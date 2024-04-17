import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import inspect

@dataclass
class GPTConfig:
    block_size: int = 256
    batch_size: int = 64
    embed_dim: int = 384
    num_layers: int = 6
    n_head: int = 6
    max_tokens: int = 1000
    dropout: int = 0.2
    vocab_size: int = 65
    bias: bool = True

class multiMaskedHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        self.qkv = nn.Linear(config.embed_dim, 3*config.embed_dim, bias=False)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(self.embed_dim, dim=-1)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q,k,v, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.tril[:,:,:T,:T]==0, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            att = self.attn_dropout(att)
            y = attn @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, 4*config.embed_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(4*config.embed_dim, config.embed_dim, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mmha = multiMaskedHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x + self.mmha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
 
        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(config.vocab_size, config.embed_dim),
            pos_emb = nn.Embedding(config.block_size, config.embed_dim),
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln = nn.LayerNorm(config.embed_dim)
        ))
        self.mapping = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer.tok_emb.weight = self.mapping.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets = None):
        B,T = x.shape
        tok_emb = self.transformer.tok_emb(x)
        pos_emb = self.transformer.pos_emb(torch.arange(0, T,  device=x.device))
        x = self.transformer.drop(tok_emb + pos_emb)

        # x = self.blocks(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln(x)
        
        if targets is not None:
            logits = self.mapping(x)
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        else:
            logits = self.mapping(x[:, [-1], :])
            loss = None
        return loss, logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad
    def generate(self, idx, max_tokens, temperature=None, top_k=None, top_p=None):
        for _ in range(max_tokens):
            idx_block = idx[:, -self.config.block_size:]

            _, logits = self(idx_block)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            idx = torch.cat((idx, next_token), dim=-1)
        return idx
