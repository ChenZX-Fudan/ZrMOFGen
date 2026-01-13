import os
from glob import glob
import shutil
import subprocess

nodes = ['Zr6']

for node in nodes:
    print(f'Now on node {node}')
    TARGET_DIR = f'data/sdf/{node}/'
    INPUT_SMILES=f'data/fragments_smi/frag_{node}.txt'
    OUTPUT_TEMPLATE=f'hMOF'
    OUT_DIR=f'data/fragments_all/{node}/'
    CORES='10'
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print('Generating molecule sdf files...')
    subprocess.run([
        'python', '-W', 'ignore', 'utils/rdkit_conf_parallel.py', 
        INPUT_SMILES, OUTPUT_TEMPLATE, '--cores', CORES
    ])
    
    for sdf_file in os.listdir('.'):
        if sdf_file.endswith('.sdf'):
            shutil.move(sdf_file, TARGET_DIR)
    
    print(f'Generating fragment and connection atom sdf files...')
    subprocess.run([
        'python', '-W', 'ignore', 'utils/prepare_dataset_parallel.py',
        '--table', INPUT_SMILES, '--sdf-dir', TARGET_DIR, 
        '--out-dir', OUT_DIR, '--template', OUTPUT_TEMPLATE, '--cores', CORES
    ])

    print(f'Filtering and merging ...')
    subprocess.run([
        'python', '-W', 'ignore', 'utils/filter_and_merge.py',
        '--in-dir', OUT_DIR, '--out-dir', OUT_DIR, 
        '--template', OUTPUT_TEMPLATE, '--number-of-files', CORES
    ])