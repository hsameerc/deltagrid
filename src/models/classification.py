"""
=============================================================================
HSRU DeltaGrid Sequence Classification Architecture
=============================================================================

Production-grade sequence classifier with masked dual-pooling (masked mean
pooling + final valid token state extraction) and non-linear SwiGLU readout.
Maintains strictly O(1) memory during chunkwise streaming over ultra-long documents.
=============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List, Union, Any, Iterable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DeltaGridConfig
from ..state import DeltaGridState
from ..outputs import DeltaGridClassificationOutput
from ..modules.block import DeltaGridBlock
from .base import _DeltaGridSequenceModel


class DeltaGridForSequenceClassification(_DeltaGridSequenceModel):
    """
    DeltaGrid Sequence Classifier for Document, Brand, Topic, and Sentiment Classification.

    Features:
      1. Tier 1 Causal Conv1D: Captures short-range n-grams, punctuation, casing, and phrasing patterns.
      2. Tier 2 Associative Delta Memory: Fast-weight Householder error-erasure memory preserving
         long-range global topic context and signature style across noisy documents.
      3. Masked Dual-Pooling: Combines masked mean pooling (global topical centroid) with the
         final valid token recurrent state (causal aggregated summary), eliminating padding leakage.
      4. SwiGLU / MLP Readout: Non-linear projection to target class logits.
      5. Constant O(1) Streaming Classification: Process arbitrary-length documents (10k-1M+ tokens)
         chunk-by-chunk without KV cache explosion or out-of-memory errors.
    """
    def __init__(
        self,
        num_classes: Optional[Union[int, DeltaGridConfig]] = None,
        vocab_size: int = 32000,
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
        classifier_dropout: float = 0.1,
        pooling_strategy: str = "dual",
        config: Optional[DeltaGridConfig] = None,
        **kwargs: Any,
    ):
        super().__init__()

        # Resolve configuration object
        if isinstance(num_classes, DeltaGridConfig):
            self.config = num_classes
        elif config is not None:
            self.config = config
        else:
            actual_classes = num_classes if num_classes is not None else kwargs.get("num_classes", 2)
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
                num_classes=actual_classes,
                classifier_dropout=classifier_dropout,
                pooling_strategy=pooling_strategy,
                **kwargs,
            )

        cfg = self.config
        self.vocab_size = cfg.vocab_size if cfg.vocab_size is not None else 32000
        self.num_classes = cfg.num_classes if cfg.num_classes is not None else 2
        self.d_model = cfg.d_model
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.num_slots = cfg.num_slots
        self.thinking_cycles = cfg.thinking_cycles
        self.pooling_strategy = cfg.pooling_strategy

        # Token embedding lookup
        self.embed = nn.Embedding(self.vocab_size, self.d_model)

        # Deep 4-tier recurrent backbone
        self.blocks = nn.ModuleList([
            DeltaGridBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_slots=self.num_slots,
                grid_dim=cfg.grid_dim,
                top_k=cfg.top_k,
                decay_init=cfg.decay_init,
                dynamic_decay=cfg.dynamic_decay,
                strict_slots=cfg.strict_slots,
                use_chunkwise=cfg.use_chunkwise,
                chunk_size=cfg.chunk_size,
            ) for _ in range(self.num_layers)
        ])

        # Layer normalization over the Pre-Norm residual stream
        self.norm = nn.LayerNorm(self.d_model)

        # Learned residual token salience gate: modulates token importance in (0, 2) prior to pooling
        self.salience_gate = nn.Linear(self.d_model, self.d_model)

        # Dimension calculation for sequence pooling
        if self.pooling_strategy == "moments":
            pooled_dim = self.d_model * 3
        elif self.pooling_strategy == "dual":
            pooled_dim = self.d_model * 2
        else:
            pooled_dim = self.d_model
        self.pooled_dim = pooled_dim

        self.dropout = nn.Dropout(cfg.classifier_dropout)

        # Non-linear classification head (Projection -> Activation -> Dropout -> Logits)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, self.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(self.d_model, self.num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes embedding, salience gate, and classification projection weights."""
        emb_std = (0.5 / math.sqrt(self.d_model)) if self.vocab_size > 256 else 0.25
        nn.init.normal_(self.embed.weight, mean=0.0, std=emb_std)

        # Zero-init salience gate so 1 + tanh(0) = 1.0 (exact identity at initialization)
        nn.init.zeros_(self.salience_gate.weight)
        nn.init.zeros_(self.salience_gate.bias)

        # Initialize linear layers in classification head
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _apply_salience(self, h: torch.Tensor) -> torch.Tensor:
        """Normalizes the residual stream and applies learned residual token salience modulation."""
        h_norm = self.norm(h)
        return h_norm * (1.0 + torch.tanh(self.salience_gate(h_norm)))

    def pool_sequence(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extracts pooled sequence-level representations (1st moment mean, 2nd moment std,
        and final valid recurrent token state) invariant to variable right-padding.

        Args:
            h (torch.Tensor): Salience-gated hidden states [batch_size, seq_len, d_model].
            attention_mask (Optional[torch.Tensor]): Attention mask tensor [batch_size, seq_len]
                where 1/True denotes valid tokens and 0/False denotes padding.

        Returns:
            torch.Tensor: Normalized pooled representation [batch_size, pooled_dim].
        """
        b, t, d = h.shape

        if attention_mask is None:
            mean_pool = h.mean(dim=1)
            var_pool = (h ** 2).mean(dim=1) - (mean_pool ** 2)
            std_pool = torch.sqrt(var_pool.clamp(min=1e-6))
            last_pool = h[:, -1, :]
        else:
            # Ensure float mask matching tensor device and precision
            mask = (attention_mask != 0).to(dtype=h.dtype)
            mask_expanded = mask.unsqueeze(-1)  # [b, t, 1]
            lens = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [b, 1]

            # 1. Masked Mean Pooling: Global 1st-moment centroid over all valid tokens
            h_masked = h * mask_expanded
            mean_pool = h_masked.sum(dim=1) / lens  # [b, d]

            # 2. Masked Standard Deviation Pooling: 2nd-moment sequence rhythm / burstiness
            sq_mean = ((h_masked ** 2).sum(dim=1) / lens)
            var_pool = sq_mean - (mean_pool ** 2)
            std_pool = torch.sqrt(var_pool.clamp(min=1e-6))  # [b, d]

            # 3. Final Valid Token State: Recurrent state at the true sequence end
            last_idx = (lens.long() - 1).clamp(min=0)  # [b, 1]
            last_idx_expanded = last_idx.unsqueeze(-1).expand(b, 1, d)  # [b, 1, d]
            last_pool = torch.gather(h, 1, last_idx_expanded).squeeze(1)  # [b, d]

        return self._combine_pools(mean_pool, last_pool, std_pool)

    def _combine_pools(
        self,
        mean_pool: torch.Tensor,
        last_pool: torch.Tensor,
        std_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combines sequence moments according to pooling_strategy."""
        if self.pooling_strategy == "moments":
            actual_std = std_pool if std_pool is not None else torch.zeros_like(mean_pool)
            return torch.cat([mean_pool, actual_std, last_pool], dim=-1)
        elif self.pooling_strategy == "dual":
            return torch.cat([mean_pool, last_pool], dim=-1)
        elif self.pooling_strategy == "mean":
            return mean_pool
        return last_pool

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
        thinking_cycles: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Uses DeltaGrid as a standalone neural sequence feature extractor, returning
        the normalized pooled representation [batch_size, pooled_dim].
        """
        h, _ = self._forward_blocks(
            self.embed(input_ids),
            states=states,
            seq_offset=seq_offset,
            thinking_cycles=thinking_cycles,
        )
        h_salient = self._apply_salience(h)
        return self.pool_sequence(h_salient, attention_mask=attention_mask)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Computes classification or regression loss from logits and target labels."""
        if labels is None:
            return None
        if self.num_classes == 1:
            if labels.dtype.is_floating_point:
                return F.mse_loss(logits.squeeze(-1), labels.float().reshape(-1))
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float().reshape(-1))
        if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1):
            return F.cross_entropy(
                logits.reshape(-1, self.num_classes),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
        state: Optional[Union[DeltaGridState, List[Any]]] = None,
        return_state: bool = True,
        thinking_cycles: Optional[int] = None,
        **kwargs: Any,
    ) -> DeltaGridClassificationOutput:
        """
        Sequence classification forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len].
            attention_mask (Optional[torch.Tensor]): Binary or boolean mask [batch_size, seq_len].
            labels (Optional[torch.Tensor]): Target labels for loss computation.
            states (Optional[Union[DeltaGridState, List[Any]]]): Recurrent states for stateful resumption.
            seq_offset (int): Absolute sequence position offset.
            state (Optional): Alias for states (trainer compatibility).
            return_state (bool): Whether to return updated recurrent states. Default: True.
            thinking_cycles (Optional[int]): Override for cognitive recurrence loops.

        Returns:
            DeltaGridClassificationOutput: Output container supporting both attribute access (.logits, .loss)
                and legacy tuple unpacking (loss, logits = model(...)).
        """
        actual_states = state if state is not None else states

        # Embed tokens and execute deep recurrent blocks
        h, new_states = self._forward_blocks(
            self.embed(input_ids),
            states=actual_states,
            seq_offset=seq_offset,
            thinking_cycles=thinking_cycles,
        )

        # Apply learned token salience gating prior to sequence moment pooling
        h_salient = self._apply_salience(h)

        # Pool sequence tokens into fixed-size normalized representation
        pooled = self.pool_sequence(h_salient, attention_mask=attention_mask)

        # Compute classification logits and optional loss
        logits = self.classifier(self.dropout(pooled))
        loss = self._compute_loss(logits, labels)

        return DeltaGridClassificationOutput(
            logits=logits,
            loss=loss,
            state=new_states if return_state else None,
            pooled=pooled,
            hidden_states=(h,),
        )

    def forward_streaming(
        self,
        chunks: Iterable[torch.Tensor],
        attention_masks: Optional[Iterable[Optional[torch.Tensor]]] = None,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        labels: Optional[torch.Tensor] = None,
        thinking_cycles: Optional[int] = None,
    ) -> DeltaGridClassificationOutput:
        """
        Processes ultra-long documents (10k-1M+ tokens) in streaming chunks with strictly constant O(1) RAM.

        Accumulates global 1st-moment (mean) and 2nd-moment (variance/std) statistics and preserves
        recurrent Householder memory states across chunks.

        Args:
            chunks (Iterable[torch.Tensor]): Sequence of token ID chunks, each [batch_size, chunk_len].
            attention_masks (Optional[Iterable[Optional[torch.Tensor]]]): Corresponding chunk attention masks.
            states (Optional): Initial recurrent state container.
            labels (Optional[torch.Tensor]): Target labels for loss computation.
            thinking_cycles (Optional[int]): Override for cognitive recurrence loops.

        Returns:
            DeltaGridClassificationOutput: Final classification predictions over the full document.
        """
        curr_states = states
        seq_offset = 0

        # Accumulators for masked mean and variance pooling across all chunks
        total_sum: Optional[torch.Tensor] = None
        total_sq_sum: Optional[torch.Tensor] = None
        total_lens: Optional[torch.Tensor] = None
        last_token_state: Optional[torch.Tensor] = None

        masks_iter = iter(attention_masks) if attention_masks is not None else None

        for chunk in chunks:
            chunk_mask = next(masks_iter) if masks_iter is not None else None
            b, t = chunk.shape

            h, curr_states = self._forward_blocks(
                self.embed(chunk),
                states=curr_states,
                seq_offset=seq_offset,
                thinking_cycles=thinking_cycles,
            )
            h_salient = self._apply_salience(h)

            if chunk_mask is None:
                chunk_sum = h_salient.sum(dim=1)
                chunk_sq_sum = (h_salient ** 2).sum(dim=1)
                chunk_cnt = torch.full((b, 1), t, dtype=h_salient.dtype, device=h_salient.device)
                last_token_state = h_salient[:, -1, :].clone()
            else:
                m = (chunk_mask != 0).to(dtype=h_salient.dtype)
                h_masked = h_salient * m.unsqueeze(-1)
                chunk_sum = h_masked.sum(dim=1)
                chunk_sq_sum = (h_masked ** 2).sum(dim=1)
                chunk_cnt = m.sum(dim=1, keepdim=True)

                has_valid = chunk_cnt.squeeze(-1) > 0
                if has_valid.any():
                    chunk_lens = chunk_cnt.clamp(min=1).long()
                    last_idx = (chunk_lens - 1).clamp(min=0).unsqueeze(-1).expand(b, 1, self.d_model)
                    cand_last = torch.gather(h_salient, 1, last_idx).squeeze(1)
                    if last_token_state is None:
                        last_token_state = cand_last
                    else:
                        last_token_state = torch.where(has_valid.unsqueeze(-1), cand_last, last_token_state)

            if total_sum is None:
                total_sum = chunk_sum
                total_sq_sum = chunk_sq_sum
                total_lens = chunk_cnt
            else:
                total_sum = total_sum + chunk_sum
                total_sq_sum = total_sq_sum + chunk_sq_sum
                total_lens = total_lens + chunk_cnt

            seq_offset += t

        if total_sum is None or total_sq_sum is None or total_lens is None:
            raise ValueError("forward_streaming received an empty sequence of chunks.")

        safe_lens = total_lens.clamp(min=1)
        mean_pool = total_sum / safe_lens
        var_pool = (total_sq_sum / safe_lens) - (mean_pool ** 2)
        std_pool = torch.sqrt(var_pool.clamp(min=1e-6))

        if last_token_state is None:
            last_token_state = mean_pool

        pooled = self._combine_pools(mean_pool, last_token_state, std_pool)
        logits = self.classifier(self.dropout(pooled))
        loss = self._compute_loss(logits, labels)

        return DeltaGridClassificationOutput(
            logits=logits,
            loss=loss,
            state=curr_states,
            pooled=pooled,
        )

    def get_optimizer_grouped_parameters(
        self,
        weight_decay: float = 0.01,
        no_decay_keywords: Tuple[str, ...] = ("decay", "beta_bias", "div_term", "clock_scale", "norm", "bias"),
    ) -> List[Dict[str, Any]]:
        """
        Constructs parameter groups exempting 1D recurrent timescales, biases, and norms from weight decay.

        Prevents erosion of long-range associative memory retention during training.

        Args:
            weight_decay (float): Weight decay coefficient for standard 2D weight matrices.
            no_decay_keywords (Tuple[str, ...]): Substrings identifying timescale and normalization parameters.

        Returns:
            List[Dict[str, Any]]: Parameter group list suitable for torch.optim.AdamW.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Saves model weights and configuration in Hugging Face compatible format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_path)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        map_location: str = "cpu",
    ) -> DeltaGridForSequenceClassification:
        """Loads sequence classifier weights and configuration from a directory."""
        load_path = Path(pretrained_model_name_or_path)
        config = DeltaGridConfig.from_pretrained(load_path)
        model = cls(config)
        weights_file = load_path / "pytorch_model.bin"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location=map_location, weights_only=True)
            model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_lm_pretrained(
        cls,
        pretrained_lm_or_path: Union[str, Path, nn.Module],
        num_classes: int = 2,
        pooling_strategy: str = "moments",
        classifier_dropout: float = 0.1,
        map_location: str = "cpu",
    ) -> DeltaGridForSequenceClassification:
        """
        Initializes a DeltaGridForSequenceClassification backbone from a pre-trained DeltaGridLM
        model instance or checkpoint directory, transferring learned embedding and recurrent weights.
        """
        if isinstance(pretrained_lm_or_path, nn.Module):
            base_cfg_dict = pretrained_lm_or_path.config.to_dict()
            src_state = pretrained_lm_or_path.state_dict()
        else:
            load_path = Path(pretrained_lm_or_path)
            base_cfg_dict = DeltaGridConfig.from_pretrained(load_path).to_dict()
            weights_file = load_path / "pytorch_model.bin"
            src_state = (
                torch.load(weights_file, map_location=map_location, weights_only=True)
                if weights_file.exists()
                else {}
            )

        base_cfg_dict["num_classes"] = num_classes
        base_cfg_dict["pooling_strategy"] = pooling_strategy
        base_cfg_dict["classifier_dropout"] = classifier_dropout
        clf_config = DeltaGridConfig.from_dict(base_cfg_dict)
        model = cls(clf_config)

        # Copy matching backbone parameters (embed.* and blocks.*)
        own_state = model.state_dict()
        compatible = {
            k: v for k, v in src_state.items()
            if k in own_state and own_state[k].shape == v.shape and not k.startswith("norm.")
        }
        own_state.update(compatible)
        model.load_state_dict(own_state)
        return model

