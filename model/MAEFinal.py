import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer


# PyTorch nn.Module definitions for the GPT-2 model

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# change the CausalSelfAttention to SelfAttention
class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Projections for query, key, value
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, context_):
        B, T, C = x.size()  # Batch, Target sequence length, Embedding size
        context = context_.unsqueeze(0).expand(B, T, C)
        _, S, _ = context.size()  # Source sequence length

        # Compute query, key, value
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.k_proj(context).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)
        v = self.v_proj(context).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)

        # Attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        # Attention output
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Final projection
        y = self.out_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


@dataclass
class MAEConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    top_loc: int = 3


class Block(nn.Module):

    def __init__(self, config, dims):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.linears = nn.ModuleList([nn.Linear(embedding.shape[1], config.n_embd) for embedding in dims])
        self.o = nn.Linear(2 * config.n_embd, config.n_embd)

    def forward(self, x, context):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_3(x))
        # context_embed0 = self.linears[0](context[0])
        # context_embed1 = self.linears[1](context[1])
        # # context_embed2 = self.linears[2](context[2])
        # context_embed = torch.cat([context_embed0, context_embed1], dim=-1)
        # context_embed = self.o(context_embed)
        # x = x + self.cross_attn(self.ln_2(x), context_embed)
        return x


class MAE(nn.Module):

    def __init__(self, dim, config, embedding=None):
        super().__init__()
        self.config = config
        # print(config.vocab_size, config.block_size)

        self.use_embedding = 0 if embedding is None else 1
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wwe=nn.Embedding(1, config.n_embd),
            wme=nn.Embedding(1, config.n_embd),
            h=nn.ModuleList(
                [Block(config, embedding if embedding is not None else -1) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, 1, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1  # don't init this one, we will tie weights
        self.apply(self._init_weights)

        if self.use_embedding == 1:
            self.wte = embedding

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02 / math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, labels_cuda, mask_cuda, FT=0):

        # step 1: value embedding: using wwe to get the value embedding
        value_embedding = self.transformer.wwe(torch.zeros_like(labels_cuda).long())

        value_embedding = value_embedding * labels_cuda.unsqueeze(-1)

        # step 2: mask embedding: using wme to get the mask embedding
        mask_embedding = self.transformer.wme(torch.zeros_like(mask_cuda).long())

        # step3 3: select the value_embedding or mask_embedding using torch.where
        embedding = torch.where(mask_cuda.unsqueeze(-1) != 0, mask_embedding, value_embedding)

        token_embedding = self.transformer.wte.weight

        x = token_embedding + embedding

        for block in self.transformer.h:
            x = block(x, self.wte)
        x = self.transformer.ln_f(x)

        predictions = self.lm_head(x).squeeze(-1)

        # print(predictions.shape, labels_cuda.shape, mask_cuda.shape)

        # calculate the MSE Loss only on  the masked tokens
        if FT == 0:
            masked_loss = torch.mean((predictions - labels_cuda) ** 2 * mask_cuda)
        else:
            real_mask = mask_cuda == 3
            masked_loss = torch.mean((predictions - labels_cuda) ** 2 * real_mask)

        return predictions, masked_loss
