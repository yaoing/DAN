import os
from shutil import copy

import pandas as pd

df = pd.read_csv('./datasets/AffectNet/affectnet.csv')

for i in range(8):
    for j in ['train','val']:
        os.makedirs(f'./datasets/AffectNet/{j}/{i}',exist_ok=True)

for i,row in df.iterrows():
    p = row['phase']
    l = row['label']
    copy(row['img_path'], f'./datasets/AffectNet/{p}/{l}')

print('convert done.')
