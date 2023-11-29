import subprocess ,os

cmd = 'python ./code/main.py --model Resnet  --gan_framework'
subprocess.call(cmd, shell=True)

cmd2 = 'python ./code/main.py --model Swin --type S'
subprocess.call(cmd2, shell=True)

