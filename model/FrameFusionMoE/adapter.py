import torch
import torch.nn as nn
import math
from collections import Counter

class FrameCrossAttention(nn.Module):
    def __init__(self, config, embed_dim, num_heads, dropout):
        super(FrameCrossAttention, self).__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # def forward(self, hidden_states, inputs_size, layer_id):
    #     """
    #     hidden_states: [B, N*(1+L), C]
    #     inputs_size: (N, L)
    #     layer_id: int
    #     """
    #     N, L = inputs_size  # Number of frames and patches
    #     bsz, tgt_len, embed_dim = hidden_states.size()

    #     # Compute query, key, value projections
    #     query_states = self.q_proj(hidden_states) * self.scale
    #     key_states = self.k_proj(hidden_states)
    #     value_states = self.v_proj(hidden_states)

    #     # Reshape for multi-head attention
    #     query_states = self._shape(query_states, tgt_len, bsz)
    #     key_states = self._shape(key_states, tgt_len, bsz)
    #     value_states = self._shape(value_states, tgt_len, bsz)

    #     # Reshape to [B*num_heads, N, 1+L, head_dim]
    #     query_states = query_states.view(bsz * self.num_heads, N, 1+L, self.head_dim)
    #     key_states = key_states.view(bsz * self.num_heads, N, 1+L, self.head_dim)
    #     value_states = value_states.view(bsz * self.num_heads, N, 1+L, self.head_dim)

    #     attn_output_list = [None] * N

    #     if layer_id % 2 == 0:
    #         # Even layer_id: process frames from 0 to N-1
    #         for i in range(N):
    #             if i == 0:
    #                 qi = query_states[:, i]  # [B*num_heads, 1+L, head_dim]
    #             else:
    #                 with torch.no_grad():
    #                     attn_output_prev = attn_output_list[i - 1]  # [B, 1+L, embed_dim]
    #                     qi = self.q_proj(attn_output_prev) * self.scale  # [B, 1+L, embed_dim]
    #                     qi = qi.view(bsz, 1+L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
    #                     qi = qi.view(bsz * self.num_heads, 1+L, self.head_dim)  # [B*num_heads, 1+L, head_dim]

    #             ki = key_states[:, i] 
    #             vi = value_states[:, i]  

    #             # Compute attention weights
    #             attn_weights_i = torch.bmm(qi, ki.transpose(1, 2))
    #             attn_weights_i = nn.functional.softmax(attn_weights_i, dim=-1)
    #             attn_probs_i = nn.functional.dropout(attn_weights_i, p=self.dropout, training=self.training)

    #             # Compute attention output
    #             attn_output_i = torch.bmm(attn_probs_i, vi)
    #             attn_output_list[i] = attn_output_i

    #             # Store previous output for the next frame
    #             attn_output_proj_i = self.out_proj(
    #                 attn_output_i.view(bsz, self.num_heads, 1+L, self.head_dim)
    #                 .permute(0, 2, 1, 3)
    #                 .reshape(bsz, 1+L, embed_dim)
    #             )
    #             attn_output_list[i] = attn_output_proj_i

    #     else:
    #         # Odd layer_id: process frames from N-1 down to 0
    #         for i in reversed(range(N)):
    #             if i == N - 1:
    #                 # Use the last frame's own query
    #                 qi = query_states[:, i]  # [B*num_heads, 1+L, head_dim]
    #             else:
    #                 # Use the next frame's attention output as the query
    #                 with torch.no_grad():
    #                     attn_output_next = attn_output_list[i + 1]
    #                     attn_output_next = attn_output_next.view(bsz, self.num_heads, 1+L, self.head_dim)
    #                     attn_output_next = attn_output_next.permute(0, 2, 1, 3).reshape(bsz, 1+L, embed_dim)
    #                     qi = self.q_proj(attn_output_next) * self.scale
    #                     qi = qi.view(bsz, 1+L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
    #                     qi = qi.view(bsz * self.num_heads, 1+L, self.head_dim)

    #             ki = key_states[:, i]
    #             vi = value_states[:, i]

    #             # Compute attention weights and output
    #             attn_weights_i = torch.bmm(qi, ki.transpose(1, 2))
    #             attn_weights_i = nn.functional.softmax(attn_weights_i, dim=-1)
    #             attn_probs_i = nn.functional.dropout(attn_weights_i, p=self.dropout, training=self.training)
    #             attn_output_i = torch.bmm(attn_probs_i, vi)
    #             attn_output_list[i] = attn_output_i

    #     # Combine outputs
    #     attn_output = torch.stack(attn_output_list, dim=1)  # [B*num_heads, N, 1+L, head_dim]
    #     attn_output = attn_output.view(bsz, self.num_heads, N, 1+L, self.head_dim)
    #     attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(bsz, N * (1 + L), embed_dim)

    #     return self.out_proj(attn_output)
    
    def forward(self, hidden_states, inputs_size, layer_id):
        """
        hidden_states: [B, N*(1+L), C]
        inputs_size: (N, L)
        layer_id: int
        """
        N, L = inputs_size  # Number of frames and patches
        bsz, tgt_len, embed_dim = hidden_states.size()

        # Handle the guided feature
        guide_states = hidden_states.view(bsz, N, 1+L, embed_dim)
        
        if layer_id % 2 == 0:
            # Forward processing: use previous frame's hidden states as guide
            guide_states = torch.roll(guide_states, shifts=1, dims=1)
            guide_states[:, 0] = hidden_states.view(bsz, N, 1+L, embed_dim)[:, 0]  # First frame keeps original
        else:
            # Backward processing: use next frame's hidden states as guide
            guide_states = torch.roll(guide_states, shifts=-1, dims=1)
            guide_states[:, -1] = hidden_states.view(bsz, N, 1+L, embed_dim)[:, -1]  # Last frame keeps original

        # Project guide states to query space after shifting
        guide_states = guide_states.view(bsz, N*(1+L), embed_dim)

        # Linear projection
        query_states = self.q_proj(guide_states) * self.scale
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._shape(query_states, tgt_len, bsz) # [bsz, num_heads, N*(1+L), head_dim]
        key_states = self._shape(key_states, tgt_len, bsz)  
        value_states = self._shape(value_states, tgt_len, bsz)

        # Reshape for attention computation [B*num_heads*N, 1+L, head_dim]
        query_states = query_states.view(bsz * self.num_heads * N, 1+L, self.head_dim)  
        key_states = key_states.view(bsz * self.num_heads * N, 1+L, self.head_dim)     
        value_states = value_states.view(bsz * self.num_heads * N, 1+L, self.head_dim)  

        # Compute inter-frame attention attention
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # [B*num_heads*N, 1+L, 1+L]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute attention output
        attn_output = torch.bmm(attn_probs, value_states)

        # Reshape back to original dimensions [bsz, N*(1+L), embed_dim]
        attn_output = attn_output.view(bsz, self.num_heads, N, 1+L, self.head_dim)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(bsz, N*(1+L), embed_dim)

        return self.out_proj(attn_output)


class LoRAAdapter(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_nums: int = 2,
                 topk: int = 2,
                 lora_dropout: float = 0.1):
        super().__init__()
        
        self.r = r
        self.lora_nums = lora_nums
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        if self.lora_nums < topk:
            self.topk = self.lora_nums
        else:
            self.topk = topk
            
        self.choose_map = torch.zeros([self.lora_nums]) # count the number of samples that each expert gets

        self.dropout = nn.Dropout(p=lora_dropout)
        self.softplus = nn.Softplus()
        
        self.noisy_gating = True
        self.is_train = True

        # LoRA matrices - one A and multiple Bs
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_Bs = nn.ModuleList([
            nn.Linear(r, out_features, bias=False) 
            for _ in range(lora_nums)
        ])
        
        # Routing components
        self.lora_route = nn.Linear(in_features, lora_nums, bias=False)
        self.w_noise = nn.Linear(in_features, lora_nums, bias=False)
        
        # Initialize weights
        self.reset_parameters()
        # self.register_gradient_hooks()

    def reset_choose_map(self):
        self.choose_map.zero_()  

    def reset_parameters(self):
        # Initialize router and noise
        nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_noise.weight)
        
        # Initialize LoRA matrices 
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        for lora_B in self.lora_Bs:
            nn.init.zeros_(lora_B.weight)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.is_train = True
        else:
            self.is_train = False

    def eval(self):
        return self.train(False)
    
    def noisy_top_k_gating(self, x, train=True, noise_epsilon=1e-2):
        clean_logits = self.lora_route(x)
        
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)  
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            
        # Get top-k gates
        top_logits, top_indices = logits.topk(min(self.topk + 1, self.lora_nums), dim=-1)
        top_k_logits = top_logits[:, :self.topk] 
        top_k_indices = top_indices[:, :self.topk]
        top_k_gates = nn.functional.softmax(top_k_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        
        # Expand gates to full size
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        return gates

    def register_gradient_hooks(self):
        self.gradient_stats = {}
        
        def hook_fn(name):
            def fn(grad):
                if grad is not None:
                    self.gradient_stats[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'zero_frac': (grad == 0).float().mean().item()
                    }
                return grad
            return fn
        
        # Register hooks for key parameters
        self.lora_route.weight.register_hook(hook_fn('route'))
        # self.lora_A.weight.register_hook(hook_fn('lora_A'))
        # for i, B in enumerate(self.lora_Bs):
        #     B.weight.register_hook(hook_fn(f'lora_B_{i}'))

    def forward(self, x: torch.Tensor, eof_index = None, task_prototype = None, task_id = None):
        if task_id ==1 and task_prototype is not None:
            lora_output = self.lora_Bs[0](self.lora_A(self.dropout(x)))
        else:       
            # MoE routing input
            if eof_index is not None:
                routing_input = x[torch.arange(x.size(0)), eof_index]
                if task_prototype is not None:
                    routing_input = routing_input + task_prototype
            else:
                routing_input = x
                
            # Get gating weights
            gates = self.noisy_top_k_gating(routing_input, self.is_train)

            # Step 2: count the frequency of each expert being selected
            if not self.is_train:
                nonzero_indices = torch.nonzero(gates) # index of selected experts
                counter = Counter(nonzero_indices[:, 1].tolist()) # count the number of samples that each expert gets
                for number, count in counter.items():
                    self.choose_map[number] = self.choose_map[number] + count

            # Apply LoRA with gating
            shared_output = self.lora_A(self.dropout(x))
            
            outputs = []
            for i in range(self.lora_nums):
                if gates[:, i].any(): 
                    expert_output = self.lora_Bs[i](shared_output)
                    outputs.append(expert_output * gates[:, i].view(-1, 1, 1))

            lora_output = sum(outputs)
                
        return lora_output * self.scaling
    
