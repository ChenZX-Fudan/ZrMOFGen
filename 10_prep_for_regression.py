import os
import shutil
import pandas as pd

# copy atom_init.json to target dir
shutil.copy(os.path.join('utils','atom_init.json'),os.path.join('newMOFs','Zr12'))
print('copied atom_init.json ...')

# generate atom_init.json to target dir
names = []
wc = []

for file in os.listdir(os.path.join('newMOFs','Zr12')):
	if '.cif' in file:
		name = file.split('.')[0]
		names.append(name)
		wc.append(0)
df = pd.DataFrame({'name':names,'wc':wc})
df.to_csv(os.path.join('newMOFs','Zr12','id_prop.csv'),header=None,index=False)
print('generated id_prop.csv ...')