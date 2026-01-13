import os
import argparse
import subprocess

nodes = ['Zr6']

# for n_atoms in [1]:
# change to the line below to reproduce paper result
for n_atoms in range(5,15):
    print(f'Sampling {n_atoms} atoms...')
    for node in nodes:
        if node != 'V':
            print(f'Now on node: {node}')
            OUTPUT_DIR = f'output/n_atoms_{n_atoms}/{node}'
            sample='20'
            os.makedirs(OUTPUT_DIR,exist_ok=True)
            if n_atoms in range(5,11):
                subprocess.run(f'python -W ignore utils/difflinker_sample_and_analyze.py --linker_size {n_atoms} --fragments data/fragments_all/{node}/hMOF_frag.sdf --model models/geom_difflinker.ckpt --output {OUTPUT_DIR} --n_samples {sample}',shell=True)
           