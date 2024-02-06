from subprocess import call
import os
from pathlib import Path

os.system('python ./code/main.py --model U_Net --type S --quant_eval_only')
os.system('python ./code/main.py --model U_Net --type M --quant_eval_only')
os.system('python ./code/main.py --model U_Net --type L --quant_eval_only')

os.system('python ./code/main.py --model ViT --type S --quant_eval_only')
os.system('python ./code/main.py --model ViT --type M --quant_eval_only')
os.system('python ./code/main.py --model Swin --type S --quant_eval_only')

os.system('python ./code/main.py --model U_Net --type L --gan_framework pix2pix --quant_eval_only')
os.system('python ./code/main.py --model ViT --type S --gan_framework pix2pix --quant_eval_only')
os.system('python ./code/main.py --model Swin --type S --gan_framework pix2pix --quant_eval_only')

#os.system('python ./code/main.py --model Diffusion --type M --quant_eval_only')
