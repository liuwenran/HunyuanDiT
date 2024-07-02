# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

# from .attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from .models import HunYuanDiTBlock

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        reference_attn=True,
        fusion_blocks="full",
        batch_size=1,
        hidden_size=1408,
        num_tokens=1024,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.fusion_blocks = fusion_blocks
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            reference_attn,
            fusion_blocks,
            batch_size=batch_size,
        )


    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        reference_attn,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        reference_attn = reference_attn
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if self.only_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    bank_fea = [   # 把self.bank模块中的bs维度 乘上 video_length
                        rearrange(
                            d.unsqueeze(1).repeat(1, video_length, 1, 1),
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank   
                    ]
                    modify_norm_hidden_states = torch.cat(
                        [norm_hidden_states] + bank_fea, dim=1
                    )
                    hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )
                    if do_classifier_free_guidance:
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(
                            hidden_states, "(b f) d c -> (b d) f c", f=video_length
                        )
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = (
                            self.attn_temp(norm_hidden_states) + hidden_states
                        )
                        hidden_states = rearrange(
                            hidden_states, "(b d) f c -> (b f) d c", d=d
                        )

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states


        def hacked_hydit_block_infer_forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
            # Long Skip Connection
            if self.skip_linear is not None:
                cat = torch.cat([x, skip], dim=-1)
                cat = self.skip_norm(cat)
                x = self.skip_linear(cat)

            # Self-Attention
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            attn_inputs = (
                self.norm1(x) + shift_msa, freq_cis_img,
            )
            x = x + self.attn1(*attn_inputs)[0]

            if MODE == "write":
                # self.bank.append(x.clone())
                self.bank = x

            if MODE == "read":
                # x = x + self.bank[0]
                x = x + self.bank
                # text_states = torch.cat([text_states] + self.bank, dim=1)

            # Cross-Attention
            cross_inputs = (
                self.norm3(x), text_states, freq_cis_img
            )
            x = x + self.attn2(*cross_inputs)[0]

            # FFN Layer
            mlp_inputs = self.norm2(x)
            x = x + self.mlp(mlp_inputs)

            return x
        
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                pass
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, HunYuanDiTBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, HunYuanDiTBlock):  ### 将原始前向传播方法进行绑定
                    module.forward = hacked_hydit_block_infer_forward.__get__(
                        module, HunYuanDiTBlock
                    )

                # module.bank = []
                module.register_buffer('bank', torch.zeros(batch_size, self.num_tokens, self.hidden_size))
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                pass
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, HunYuanDiTBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, HunYuanDiTBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            start_ind = len(reader_attn_modules) // 2 - len(writer_attn_modules) // 2
            reader_attn_modules_to_update = reader_attn_modules[start_ind:start_ind + len(writer_attn_modules)]
            for r, w in zip(reader_attn_modules_to_update, writer_attn_modules):
                # r.bank = [v.clone().to(dtype) for v in w.bank] 
                r.bank = w.bank
                # w.bank.clear()

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                pass
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, HunYuanDiTBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.zero_()
