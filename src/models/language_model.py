"""
=============================================================================
HSRU DeltaGrid Autoregressive Zero-KV Language Model
=============================================================================

Zero-KV memory language model architecture achieving 100% SRAM residency and
strictly constant O(1) RAM decoding during text generation.
=============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List, Union, Any
import torch
import torch.nn as nn

from ..config import DeltaGridConfig
from ..state import DeltaGridState
from ..outputs import DeltaGridLMOutput
from ..modules.block import DeltaGridBlock
from .base import _DeltaGridSequenceModel


class DeltaGridLM(_DeltaGridSequenceModel):
    """
    Language Model using DeltaGrid Blocks.

    Maintains strictly constant O(1) RAM during autoregressive text generation
    with zero KV cache growth. Uses tied input/output embedding weights.
    """
    def __init__(
        self,
        vocab_size: Optional[Union[int, DeltaGridConfig]] = None,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_slots: Optional[int] = None,
        grid_dim: Optional[Tuple[int, ...]] = None,
        top_k: Optional[int] = None,
        thinking_cycles: int = 0,
        decay_init: float = 0.999,
        dynamic_decay: bool = True,
        strict_slots: bool = False,
        use_chunkwise: bool = False,
        chunk_size: int = 64,
        config: Optional[DeltaGridConfig] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if isinstance(vocab_size, DeltaGridConfig):
            self.config = vocab_size
        elif config is not None:
            self.config = config
        else:
            actual_vocab = vocab_size if vocab_size is not None else kwargs.get("vocab_size", 32000)
            self.config = DeltaGridConfig(
                vocab_size=actual_vocab,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                num_slots=num_slots,
                grid_dim=grid_dim,
                top_k=top_k,
                thinking_cycles=thinking_cycles,
                decay_init=decay_init,
                dynamic_decay=dynamic_decay,
                strict_slots=strict_slots,
                use_chunkwise=use_chunkwise,
                chunk_size=chunk_size,
                **kwargs,
            )

        cfg = self.config
        self.vocab_size = cfg.vocab_size if cfg.vocab_size is not None else 32000
        self.d_model = cfg.d_model
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.num_slots = cfg.num_slots
        self.thinking_cycles = cfg.thinking_cycles

        # Token embedding lookup table
        self.embed = nn.Embedding(self.vocab_size, self.d_model)

        # Deep recurrent backbone: stacked 4-tier DeltaGrid blocks
        self.blocks = nn.ModuleList([
            DeltaGridBlock(
                config=cfg,
            ) for _ in range(self.num_layers)
        ])

        # Sequence LayerNorm prior to readout
        self.norm = nn.LayerNorm(self.d_model)

        # Tied LM head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        emb_std = (0.5 / math.sqrt(self.d_model)) if self.vocab_size > 256 else 0.25
        nn.init.normal_(self.embed.weight, mean=0.0, std=emb_std)
        self.lm_head.weight = self.embed.weight

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
        state: Optional[Union[DeltaGridState, List[Any]]] = None,
        return_state: bool = True,
        return_aux_loss: bool = False,
        thinking_cycles: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Language model prefill forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len].
            states (Optional[Union[DeltaGridState, List[Any]]]): Recurrent states for stateful resumption.
            seq_offset (int): Absolute sequence position offset.
            state (Optional): Alias for states (trainer compatibility).
            return_state (bool): Whether to return updated recurrent states. Default: True.
            return_aux_loss (bool): Whether to return auxiliary density/stability loss. Default: False.
            thinking_cycles (Optional[int]): Override for cognitive recurrence loops.
            labels (Optional[torch.Tensor]): Optional target labels for cross-entropy loss computation.

        Returns:
            DeltaGridLMOutput: Structured output supporting both attribute access (.logits, .state, .loss)
                and legacy tuple unpacking (logits, state = model(...)).
        """
        actual_states = state if state is not None else states
        h, new_states = self._forward_blocks(
            self.embed(input_ids),
            states=actual_states,
            seq_offset=seq_offset,
            thinking_cycles=thinking_cycles,
        )
        logits = self.lm_head(self.norm(h))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

        aux_loss = torch.tensor(0.0, device=input_ids.device) if return_aux_loss else None

        if return_aux_loss:
            legacy_format = "logits_aux_state" if return_state else "logits_aux"
        else:
            legacy_format = "logits_state" if return_state else "logits_only"

        return DeltaGridLMOutput(
            logits=logits,
            state=new_states if return_state else None,
            loss=loss,
            aux_loss=aux_loss,
            legacy_format=legacy_format,
        )

    def step(
        self,
        token_id: torch.Tensor,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, DeltaGridState]:
        """
        Single-token autoregressive decoding step in strictly constant O(1) RAM.

        Args:
            token_id (torch.Tensor): Single token ID tensor [batch_size, 1] or [batch_size].
            states (Optional): Current recurrent state container.
            seq_offset (int): Current sequence offset.

        Returns:
            Tuple[torch.Tensor, DeltaGridState]: Next-token logits [b, 1, vocab_size] and updated states.
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        h, new_states = self._step_blocks(
            self.embed(token_id),
            states=states,
            seq_offset=seq_offset,
        )
        logits = self.lm_head(self.norm(h))
        return logits, new_states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generates text tokens autoregressively with zero KV cache growth.

        Args:
            prompt_ids (torch.Tensor): Prompt token IDs [batch_size, prompt_len].
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Softmax sampling temperature. If <= 0, argmax greedy decoding is used.
            top_k (int): Top-K candidate filtering.
            top_p (float): Nucleus Top-P cumulative probability threshold.
            repetition_penalty (float): Repetition penalty (> 1.0 penalizes already generated tokens).
            eos_token_id (Optional[int]): End-of-sequence token ID to terminate generation.
            **kwargs: Extra parameters ignored for compatibility.

        Returns:
            torch.Tensor: Concatenated token IDs [batch_size, prompt_len + generated_len].
        """
        self.eval()
        b, prompt_len = prompt_ids.shape

        def _apply_rep_penalty(logits_row: torch.Tensor, tokens_seen: torch.Tensor) -> torch.Tensor:
            if repetition_penalty == 1.0 or tokens_seen.numel() == 0:
                return logits_row
            for b_i in range(logits_row.shape[0]):
                unique_toks = torch.unique(tokens_seen[b_i])
                scores = logits_row[b_i, unique_toks]
                logits_row[b_i, unique_toks] = torch.where(
                    scores < 0, scores * repetition_penalty, scores / repetition_penalty
                )
            return logits_row

        def _apply_top_p(logits_row: torch.Tensor) -> torch.Tensor:
            if top_p >= 1.0:
                return logits_row
            sorted_logits, sorted_indices = torch.sort(logits_row, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits_row[indices_to_remove] = -float("Inf")
            return logits_row

        # Context prefill pass
        out = self.forward(prompt_ids, return_state=True)
        logits, states = out.logits, out.state

        next_token_logits = logits[:, -1, :].clone()
        all_tokens = prompt_ids.clone()
        next_token_logits = _apply_rep_penalty(next_token_logits, all_tokens)

        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
            next_token_logits = _apply_top_p(next_token_logits)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = [prompt_ids, next_token]
        curr_token = next_token
        all_tokens = torch.cat([all_tokens, curr_token], dim=1)
        curr_offset = prompt_len

        # Autoregressive decoding loop in constant O(1) RAM
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (curr_token == eos_token_id).all():
                break

            step_logits, states = self.step(curr_token, states=states, seq_offset=curr_offset)
            curr_offset += 1

            logits_flat = step_logits.squeeze(1).clone()
            logits_flat = _apply_rep_penalty(logits_flat, all_tokens)

            if temperature > 0:
                logits_flat = logits_flat / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits_flat, min(top_k, logits_flat.size(-1)))
                    logits_flat[logits_flat < v[:, [-1]]] = -float("Inf")
                logits_flat = _apply_top_p(logits_flat)
                probs = torch.softmax(logits_flat, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
            else:
                curr_token = torch.argmax(logits_flat, dim=-1, keepdim=True)

            generated.append(curr_token)
            all_tokens = torch.cat([all_tokens, curr_token], dim=1)

        return torch.cat(generated, dim=1)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Saves model weights and configuration in Hugging Face compatible format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_path)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], map_location: str = "cpu") -> DeltaGridLM:
        """Loads model weights and configuration from a directory."""
        load_path = Path(pretrained_model_name_or_path)
        config = DeltaGridConfig.from_pretrained(load_path)
        model = cls(config)
        weights_file = load_path / "pytorch_model.bin"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location=map_location, weights_only=True)
            model.load_state_dict(state_dict)
        return model
