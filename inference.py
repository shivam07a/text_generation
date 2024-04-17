import os
import pickle
from contextlib import nullcontext
import torch
from model_rotationalPositional_embedding import GPTConfig, GPT

max_new_tokens = 50
device = 'cuda'
seed = 1337
dtype = 'float16'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

checkpoint = torch.load('checkpoints/ckpt.pt', map_location=device)
gptconfig = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconfig)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
print(f"loading model ...")
model.load_state_dict(state_dict)
model.eval()
model.to(device)
# torch.compile(model)
print(f"model loaded")

with open('./data/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def argument_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='The quick brown fox jumps over the lazy dog')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    max_new_tokens = args.max_new_tokens
    prompt = args.prompt
    encoded_prompt_tokens = encode(prompt)
    encoded_prompt_tokens_tensor = torch.tensor(encoded_prompt_tokens, dtype=torch.long, device=device).view(1,-1)

    with torch.no_grad():
        with ctx:
            print(f"generating texts ...")
            y = model.generate(encoded_prompt_tokens_tensor, max_new_tokens, temperature=0.9, top_k=5)
            print(f'generated: {decode(y[0].tolist())}')

if __name__ == '__main__':
    main()