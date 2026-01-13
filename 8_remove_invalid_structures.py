import os
import shutil
from tqdm import tqdm
import multiprocessing as mproc
from pymatgen.core.structure import Structure
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module="pymatgen.io.cif")

os.makedirs(os.path.join('newMOFs','MOFs_invalid'),exist_ok=True)

NCPUS = int(0.9*os.cpu_count())

print(f'Number of CPUs: {NCPUS}')

def exam_cif(mof):
    try:
        Structure.from_file(os.path.join(os.path.join('newMOFs','Zr12'),mof))
    except:
        print(f'removed {mof}')
        shutil.move(os.path.join('newMOFs','Zr12',mof),os.path.join('newMOFs','MOFs_invalid'))
if __name__ == '__main__':
    with mproc.Pool(NCPUS) as mp: 
        mp.map_async(exam_cif,os.listdir(os.path.join('newMOFs','Zr12'))).get()