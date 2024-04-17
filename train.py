
import math
import os
import time
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# from distributed_genai.model_rotationalPositional_embedding import GPTConfig, GPT
from model import GPTConfig, GPT

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

eval_per_steps = 200
avg_loss_steps = 100

batch_size = 64
block_size = 256

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# adamw optimizer
lr = 1e-3
total_iters = 5001
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5001
min_lr = 1e-4

backend = 'nccl'
gradient_accumulation_steps = 4

best_val_loss = 1e9
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} 
# -----------------------------------------------------------------------------

init_process_group(backend=backend)
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0
seed_offset = ddp_rank

assert gradient_accumulation_steps % ddp_world_size == 0
gradient_accumulation_steps //= ddp_world_size

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

train_data = np.memmap('./data/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('./data/val.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

meta_vocab_size = pickle.load(open('./data/meta.pkl','rb'))['vocab_size']
model_args = dict()
model_args['vocab_size'] = meta_vocab_size
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)


# =======================================================================
# with open('../distributed_torch/meta.pkl', 'rb') as f:
#     meta = pickle.load(f)

# stoi, itos = meta['stoi'], meta['itos']
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])

# prompt_tokens = 'The quick brown fox jumps over the lazy dog'
# encoded_prompt_tokens = encode(prompt_tokens)
# encoded_prompt_tokens_tensor = torch.tensor(encoded_prompt_tokens, dtype=torch.long, device=device).view(1,-1)

# max_new_tokens = 20
# y = model.generate(encoded_prompt_tokens_tensor, max_new_tokens)
# print(f'generated: {decode(y[0].tolist())}')

# =======================================================================

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type)
checkpoint = None
# print(f"Compiling the model ...")
# model = torch.compile(model)
# print(f"Model compiled")
model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss(model):
    loss_dic = {}
    model.eval()
    for split in ['train','val']:
        loss_per_iter = torch.zeros(avg_loss_steps)
        for i in range(avg_loss_steps):
            xb, yb = get_batch(split)
            with ctx:
                loss, _ = model(xb,yb)
            loss_per_iter[i] = loss.item()

        loss_dic[split] = loss_per_iter.mean()
    model.train()
    return loss_dic

def get_lr(it):
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

out_dir = 'checkpoints'
raw_model = model.module
if device == 'cuda:0':
    os.makedirs(out_dir, exist_ok=True)

### Training
for iter in range(total_iters):
    lr = get_lr(iter) if decay_lr else lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (iter % eval_per_steps == 0) and (device == 'cuda:0'):
        loss = estimate_loss(model)
        print(f"Iter: {iter} | TotalTime : {(time.time() - start_time):.1f} | TrainLoss : {loss['train']:.2f} | ValLoss : {loss['val']:.2f}")

        if iter > 0:
            if loss['val'] < best_val_loss:
                best_val_loss = loss['val']
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter': iter,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print(f"Saving model with validation loss: {best_val_loss:.2f}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch("train")
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            loss, _ = model(xb, yb)
            loss = loss / gradient_accumulation_steps    
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

destroy_process_group()

# TORCHDYNAMO_REPRO_AFTER="aot" torchrun --standalone --nproc_per_node=4 train.py