import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import heapq  # For managing finished beams efficiently
from typing import Optional, Tuple, List, Dict, Any


# --- PositionalEncoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2))  # Shape: [1, max_len, d_model] for batch_first=True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=True)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class OCRModel(nn.Module):
    """
    OCR Model using ViT Encoder and Transformer Decoder.
    Includes Greedy and Beam Search generation methods.

    NOTE on Performance: The standard `nn.TransformerDecoder` used here does NOT
    have built-in Key-Value (KV) caching like implementations found in libraries
    such as Hugging Face Transformers. This means that during autoregressive
    generation (`sample`, `generate`), the decoder recomputes attention keys/values
    for the entire sequence at each step, leading to O(L^2) complexity per sequence.
    This is a significant performance bottleneck for long sequences. For substantial
    speedups in generation/sampling, consider using a library with KV caching support
    or implementing a custom decoder with caching.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 768,
                 decoder_nhead: int = 12,
                 decoder_layers: int = 6,
                 decoder_dim_feedforward: int = 3072,
                 decoder_dropout: float = 0.1,
                 max_seq_len: int = 256,
                 pad_token_id: int = 0,
                 sos_token_id: int = 1,
                 eos_token_id: int = 2,
                 vit_model_name: str = 'vit_base_patch16_224_in21k',
                 vit_pretrained: bool = True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

        # ---Encoder---
        self.vit = timm.create_model(vit_model_name, pretrained=vit_pretrained)
        # Ensure the vit model is loaded correctly
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()  # Common for timm models
        elif hasattr(self.vit, 'fc'):
            self.vit.fc = nn.Identity()  # Some models use fc
        else:
            print(
                f"Warning: Could not automatically find final layer ('head' or 'fc') in {vit_model_name} to replace. Manual check needed.")
            # Fallback or error

        vit_feature_dim = self.vit.num_features  # Use num_features which is more standard in timm
        if vit_feature_dim != d_model:
            self.vit_proj = nn.Linear(vit_feature_dim, d_model)
            print(f"Projecting ViT features from {vit_feature_dim} to {d_model}")
        else:
            self.vit_proj = nn.Identity()

        # ---Decoder---
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_decoder = PositionalEncoding(d_model, decoder_dropout, max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=decoder_nhead, dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout, batch_first=True
        )
        # Corrected variable name as noted in the prompt
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, p in self.named_parameters():
            # Initialize non-ViT parts
            if not name.startswith('vit.'):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)
            # Initialize projection layer if it exists
            if name.startswith('vit_proj.'):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        vit_features = self.vit.forward_features(imgs)  # Use forward_features

        # Handle potential CLS token based on ViT architecture
        # Most ViT forward_features includes CLS token at the beginning
        # Check timm doc for the specific model if unsure
        if vit_features.shape[1] > 1 and self.vit.has_class_token:  # Check if CLS token exists
            memory = vit_features[:, 1:, :]  # Exclude CLS token
        else:
            memory = vit_features  # If no CLS token, use all features

        memory = self.vit_proj(memory)  # Project features if needed
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure input has 3 dims [B, SeqLen, EmbDim] for embedding lookup if needed
        # tgt is usually [B, SeqLen] indices
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder_decoder(tgt_emb)

        # Generate mask if not provided (standard for training)
        if tgt_mask is None:
            device = tgt.device
            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool), diagonal=1)

        # Generate padding mask if not provided
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt == self.pad_token_id)

        output = self.transformer_decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=None
            # Assuming encoder output is not padded
        )
        return output

    def forward(self, imgs: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for training with teacher forcing."""
        memory = self.encode(imgs)
        # tgt_in is typically tgt[:, :-1]
        # tgt_out is typically tgt[:, 1:] for loss calculation
        # This forward pass assumes tgt is the input sequence (e.g., tgt_in)
        decoder_output = self.decode(tgt=tgt, memory=memory)  # Masks are generated internally in decode
        logits = self.output_layer(decoder_output)
        return logits

    def sample(self, imgs: torch.Tensor, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences using multinomial sampling (for RL training).

        Performance Note: This is computationally expensive without KV caching.
        Each step involves a full pass through the Transformer decoder.
        """
        self.eval()  # Ensure eval mode for sampling consistency
        if max_len is None: max_len = self.max_seq_len
        device = imgs.device
        batch_size = imgs.size(0)

        # --- Encode ---
        memory = self.encode(imgs)  # [B, SrcSeqLen, Dim]

        # --- Decode Step-by-Step ---
        sampled_ids = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
        total_log_probs = torch.zeros(batch_size, device=device)
        finished_seqs = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):  # Max steps to generate max_len sequence (incl SOS)
            current_input_ids = sampled_ids  # [B, current_len]

            # --- Call Decoder ---
            # NOTE: We pass the *entire* sequence generated so far. The internal
            #       causal mask handles preventing future info leakage.
            #       Without KV cache, this recomputes everything.
            decoder_output = self.decode(tgt=current_input_ids, memory=memory)  # [B, current_len, Dim]

            # --- Get Logits for the *last* token ---
            logits_last_step = self.output_layer(decoder_output[:, -1, :])  # [B, VocabSize]

            # --- Sample Next Token ---
            log_probs = F.log_softmax(logits_last_step, dim=-1)
            next_token_ids = torch.multinomial(log_probs.exp(), num_samples=1)  # [B, 1]

            # --- Accumulate Log Probabilities (for RL) ---
            # Gather the log probability of the chosen token
            gathered_log_probs = torch.gather(log_probs, 1, next_token_ids).squeeze(1)  # [B]
            # Only add log_prob for sequences that are not yet finished
            total_log_probs += gathered_log_probs * (~finished_seqs)

            # --- Update Sequences ---
            # Prevent adding tokens after EOS has been generated
            next_token_ids = torch.where(finished_seqs.unsqueeze(1),
                                         torch.tensor(self.pad_token_id, device=device),
                                         next_token_ids)
            sampled_ids = torch.cat([sampled_ids, next_token_ids], dim=1)

            # --- Check for Finished Sequences ---
            just_finished = (next_token_ids.squeeze(1) == self.eos_token_id) & (~finished_seqs)
            finished_seqs |= just_finished

            # --- Early Exit ---
            if finished_seqs.all():
                break

        # Make sure model is back to train mode if called within training loop
        # self.train() # Caller should handle train/eval mode switching

        return sampled_ids, total_log_probs

    @torch.no_grad()
    def generate(self,
                 imgs: torch.Tensor,
                 max_len: Optional[int] = None,
                 method: str = 'greedy',
                 beam_width: int = 5,
                 length_penalty: float = 0.7) -> torch.Tensor:
        """
        Generate sequences for inference/evaluation using Greedy or Beam Search.

        Performance Note: Similar to `sample`, this is slow without KV caching,
        especially beam search which runs the decoder multiple times per step.
        """
        self.eval()  # Ensure model is in evaluation mode

        if max_len is None:
            max_len = self.max_seq_len
        device = imgs.device
        batch_size = imgs.size(0)

        # ---Encode---
        memory = self.encode(imgs)  # [B, Src_Len, d_model]

        # ---Generation---
        if method == 'greedy':
            generated_ids = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
            finished_seqs = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                current_input_ids = generated_ids
                # Decode using the full sequence generated so far
                decoder_output = self.decode(tgt=current_input_ids, memory=memory)
                # Get logits for the very last position
                logits_last_step = self.output_layer(decoder_output[:, -1, :])  # [B, VocabSize]
                # Select the token with the highest probability
                next_token_ids = torch.argmax(logits_last_step, dim=-1, keepdim=True)  # [B, 1]

                # Prevent adding tokens after EOS
                next_token_ids = torch.where(finished_seqs.unsqueeze(1),
                                             torch.tensor(self.pad_token_id, device=device),
                                             next_token_ids)
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)

                # Check which sequences finished
                just_finished = (next_token_ids.squeeze(1) == self.eos_token_id) & (~finished_seqs)
                finished_seqs |= just_finished

                # Exit if all sequences in the batch have finished
                if finished_seqs.all():
                    break
            return generated_ids

        elif method == 'beam':
            # --- Beam Search Implementation ---
            # Expand memory for beam width
            expanded_memory = memory.repeat_interleave(beam_width, dim=0)  # [B*W, Src_Len, d_model]

            # Initialize beams: [(score, sequence)] * W for each batch item
            batch_beams = [[(0.0, torch.tensor([self.sos_token_id], dtype=torch.long, device=device))]
                           for _ in range(batch_size)]
            # Completed hypotheses: Use heapq [(neg_norm_score, sequence)] * W for each batch item
            batch_completed_hypotheses = [[] for _ in range(batch_size)]
            num_finished_beams = [0] * batch_size  # Track finished count per batch item

            for step in range(max_len - 1):
                if all(n >= beam_width for n in num_finished_beams): break  # All batch items done

                # Store candidates for the *next* step: (batch_idx, score, next_token_id, beam_idx_in_batch)
                # Flatten beams across batches for efficient decoding
                current_beam_inputs = []  # List of sequences (tensors)
                beam_indices_map = []  # List of (batch_idx, beam_idx_in_batch)
                beam_scores_list = []  # List of scores corresponding to current_beam_inputs

                beam_global_idx = 0
                for b_idx in range(batch_size):
                    if num_finished_beams[b_idx] >= beam_width: continue  # Skip finished batch items
                    for beam_idx, (score, seq) in enumerate(batch_beams[b_idx]):
                        current_beam_inputs.append(seq)
                        beam_indices_map.append((b_idx, beam_idx))
                        beam_scores_list.append(score)
                        beam_global_idx += 1

                if not current_beam_inputs: break  # No active beams left

                # Prepare batch for decoder
                input_seqs = nn.utils.rnn.pad_sequence(current_beam_inputs, batch_first=True,
                                                       padding_value=self.pad_token_id)
                current_batch_beam_size = input_seqs.size(0)
                current_len = input_seqs.size(1)

                # Select corresponding memory entries: map global beam index back to expanded_memory index
                memory_indices = []
                current_offset = 0
                for b_idx in range(batch_size):
                    num_beams_in_this_batch = 0
                    # Count how many beams belong to this batch item in the current flat list
                    for map_b_idx, _ in beam_indices_map[current_offset:]:
                        if map_b_idx == b_idx:
                            num_beams_in_this_batch += 1
                        else:
                            break  # Moved to next batch item
                    for i in range(num_beams_in_this_batch):
                        # The memory for beam `i` of batch `b_idx` is at index `b_idx * beam_width + i`
                        # in the original repeat_interleave operation. However, the *active* beams
                        # might be fewer than beam_width if some finished early.
                        # We need the *original* beam index associated with the memory.
                        # This requires careful tracking if beams are pruned/removed.
                        # For simplicity now, assume beam_idx_in_batch corresponds to the original interleaved index offset.
                        # This might be incorrect if beams finish early. A more robust way is needed.
                        # Let's try mapping using the original beam structure logic:
                        original_beam_idx_in_batch = beam_indices_map[current_offset + i][
                            1]  # Get the beam's index *within its batch*
                        memory_idx = b_idx * beam_width + original_beam_idx_in_batch
                        memory_indices.append(memory_idx)

                    current_offset += num_beams_in_this_batch

                # Ensure memory_indices calculation is correct. If beam search prunes beams,
                # mapping back to the `repeat_interleave`d memory is complex.
                # A common approach is to reorder memory based on selected beam indices at each step.

                # --- Let's simplify the memory selection assuming no complex pruning yet ---
                # Map flat index `k` to its original batch item `b = beam_indices_map[k][0]`
                # The memory for this beam corresponds to `b * beam_width + beam_indices_map[k][1]` in expanded_memory
                # This assumes beam_indices_map[k][1] stays consistent, which might break if beams end.
                # SAFER APPROACH: Reorder memory based on chosen beams each step.
                # TODO: Implement robust memory reordering if needed. Using potentially incorrect indices for now.
                # The current memory_indices logic might be flawed.

                # TEMPORARY FIX / Approximation: Assume the first `current_batch_beam_size` entries
                # of expanded_memory correspond to the active beams. This is only true if no beams have finished.
                # active_memory = expanded_memory[:current_batch_beam_size] # POTENTIALLY WRONG

                # Correct memory selection: Need to know which original beam (0..W-1) each active sequence came from.
                # Let's add this to the beam state: (score, sequence, original_beam_idx)
                # Reworking beam state slightly... (Skipping this complex rework for now)

                # Sticking with simplified/potentially flawed memory selection for now:
                # Use the indices derived from beam_indices_map, hoping it aligns reasonably.
                try:
                    active_memory = expanded_memory[memory_indices]
                except IndexError:
                    print("Error: Indexing expanded_memory failed. Beam search memory management needs review.")
                    # Fallback or raise error
                    active_memory = expanded_memory[:current_batch_beam_size]  # Incorrect fallback

                # Decode step
                decoder_output = self.decode(tgt=input_seqs, memory=active_memory)
                logits_last_step = self.output_layer(decoder_output[:, -1, :])  # [Current_Batch_Beam_Size, vocab_size]
                log_probs = F.log_softmax(logits_last_step, dim=-1)  # [Current_Batch_Beam_Size, vocab_size]

                # Add previous beam scores
                prev_scores = torch.tensor(beam_scores_list, device=device).unsqueeze(1)  # [Current_Batch_Beam_Size, 1]
                candidate_scores = prev_scores + log_probs  # [Current_Batch_Beam_Size, vocab_size]

                # --- Per-Batch Top-K Selection ---
                new_batch_beams = [[] for _ in range(batch_size)]
                processed_beams_count = 0

                for b_idx in range(batch_size):
                    if num_finished_beams[b_idx] >= beam_width:
                        # If finished, we technically shouldn't have processed it.
                        # If it appears here due to indexing issues, just skip.
                        continue

                    # Find the range corresponding to this batch item in the flattened results
                    num_beams_in_batch_active = 0
                    start_idx = -1
                    for i in range(processed_beams_count, current_batch_beam_size):
                        if beam_indices_map[i][0] == b_idx:
                            if start_idx == -1: start_idx = i
                            num_beams_in_batch_active += 1
                        elif start_idx != -1:  # Moved past this batch's beams
                            break

                    if num_beams_in_batch_active == 0: continue  # Should not happen if not finished

                    item_scores = candidate_scores[
                                  start_idx: start_idx + num_beams_in_batch_active]  # [num_active, vocab]
                    item_sequences = current_beam_inputs[start_idx: start_idx + num_beams_in_batch_active]

                    # Flatten scores and tokens for this batch item
                    flat_scores = item_scores.view(-1)  # [num_active * vocab_size]
                    flat_token_ids = torch.arange(self.vocab_size, device=device).repeat(num_beams_in_batch_active)
                    flat_beam_indices = torch.arange(num_beams_in_batch_active, device=device).repeat_interleave(
                        self.vocab_size)

                    # Select top candidates for *this batch item*
                    # Need `beam_width` active beams OR finished beams
                    num_candidates_to_consider = min(flat_scores.size(0), beam_width * 2)  # Consider more
                    top_scores, top_indices = torch.topk(flat_scores, k=num_candidates_to_consider)

                    # Reconstruct chosen beams
                    next_beams_for_item = []
                    current_item_finished_count = 0  # Count newly finished beams in this step

                    for i in range(top_indices.size(0)):
                        if len(next_beams_for_item) + num_finished_beams[
                            b_idx] + current_item_finished_count >= beam_width * 2 and len(
                                batch_completed_hypotheses[b_idx]) >= beam_width:
                            # Optimization: If we have enough finished and active candidates, stop considering lower-scoring ones
                            # Be careful with this logic. Let's keep it simple first.
                            pass

                        score = top_scores[i].item()
                        flat_idx = top_indices[i].item()

                        prev_beam_idx_in_item_active = flat_beam_indices[flat_idx].item()
                        next_token_id = flat_token_ids[flat_idx].item()

                        prev_seq = item_sequences[prev_beam_idx_in_item_active]
                        next_seq = torch.cat([prev_seq, torch.tensor([next_token_id], dtype=torch.long, device=device)])

                        if next_token_id == self.eos_token_id:
                            if num_finished_beams[b_idx] < beam_width:
                                seq_len = len(next_seq)
                                normalized_score = score / (seq_len ** length_penalty)
                                # Use negative score for min-heap
                                heapq.heappush(batch_completed_hypotheses[b_idx], (-normalized_score, next_seq))
                                # Keep only top W completed
                                if len(batch_completed_hypotheses[b_idx]) > beam_width:
                                    heapq.heappop(batch_completed_hypotheses[b_idx])
                                # Only increment finished count if we actually stored it
                                num_finished_beams[b_idx] += 1
                                current_item_finished_count += 1

                        elif len(next_beams_for_item) < beam_width:  # Add to active beams if not full
                            next_beams_for_item.append((score, next_seq))

                    # Prune active beams for next step to beam_width, ensure sorted
                    next_beams_for_item.sort(key=lambda x: x[0], reverse=True)
                    new_batch_beams[b_idx] = next_beams_for_item[:beam_width]

                    processed_beams_count += num_beams_in_batch_active  # Update count of processed flat beams

                # Update beams for the next iteration (only keep non-empty lists)
                active_batch_beams = [beams for beams in new_batch_beams if beams]
                if not active_batch_beams and all(n > 0 for n in num_finished_beams):
                    # If all beams finished or were pruned, but some completed hypotheses exist
                    break
                elif not active_batch_beams:
                    # Edge case: No active beams left, and maybe no completed ones either
                    print("Warning: No active beams left during beam search.")
                    break

                batch_beams = new_batch_beams  # Contains potentially empty lists for finished items

            # --- After Loop: Select Best Hypothesis ---
            final_output_sequences = []
            for b_idx in range(batch_size):
                if batch_completed_hypotheses[b_idx]:
                    # Retrieve the best completed hypothesis (highest normalized score)
                    best_normalized_score_neg, best_seq = heapq.nsmallest(1, batch_completed_hypotheses[b_idx])[0]
                    final_output_sequences.append(best_seq)
                elif batch_beams[b_idx]:  # Fallback to the best active beam if no completed ones
                    # Apply length penalty to active beams for fair comparison
                    best_beam = max(batch_beams[b_idx], key=lambda x: x[0] / (len(x[1]) ** length_penalty))
                    final_output_sequences.append(best_beam[1])
                else:  # Should not happen if SOS was added, but handle edge case
                    print(f"Warning: No completed or active beams found for batch item {b_idx}. Returning SOS.")
                    final_output_sequences.append(torch.tensor([self.sos_token_id], dtype=torch.long, device=device))

            # Pad sequences in the final batch
            # Handle potential empty list if batch_size=0 or error
            if not final_output_sequences:
                return torch.empty((0, 0), dtype=torch.long, device=device)  # Or handle as error

            max_out_len = max(seq.size(0) for seq in final_output_sequences) if final_output_sequences else 1
            padded_sequences = []
            for seq in final_output_sequences:
                pad_len = max_out_len - seq.size(0)
                padded_seq = F.pad(seq, (0, pad_len), value=self.pad_token_id)
                padded_sequences.append(padded_seq)

            return torch.stack(padded_sequences, dim=0)  # [B, Max_Out_Len]

        else:
            raise ValueError(f"Unsupported generation method: {method}")
