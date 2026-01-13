import os
import subprocess
import itertools
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load and clean data
df_info = pd.read_csv('data/Zr6_hMOF.csv')
df_info = df_info.dropna()  # Drop entries containing 'NaN'
df_info = df_info[df_info.CO2_capacity_001 > 0]  # Only keep entries with positive CO2 working capacity
df_info = df_info[~df_info.MOFid.str.contains('ERROR')]  # Drop entries with error
df_info = df_info[~df_info.MOFid.str.contains('NA')]  # Drop entries with NA

# Create necessary folders
os.makedirs('data/conformers', exist_ok=True)
os.makedirs('data/data_by_node', exist_ok=True)
os.makedirs('data/fragments_smi', exist_ok=True)

# Define the target node (Zr6 cluster with specific connectivity)
target_node = '[O]12[Zr]34[O]5[Zr]62[O]2[Zr]71[O]4[Zr]14[O]3[Zr]35[O]6[Zr]2([O]71)[O]43'
node_name = 'Zr6'

# Filter data for the specific node
print(f"Filtering data for node: {target_node}")
df_target_node = df_info[df_info['node'] == target_node]

# Save the filtered data to CSV
output_path = f'data/data_by_node/{node_name}.csv'
df_target_node.to_csv(output_path, index=False)
print(f"Saved {len(df_target_node)} records to {output_path}")

# Process the filtered data
print(f'Now processing node {node_name} ... ')

# Check if the file was created successfully
if not os.path.exists(output_path):
    print(f"Error: File {output_path} was not created!")
    exit()

# Load the filtered data
df = pd.read_csv(output_path)
print(f"Loaded {len(df)} MOF records for node {node_name}")

# MOFs with high working capacity at 0.1 bar (wc >= 1.5 mmol/g)
df_high_wc = df[df['CO2_capacity_01'] >= 1.5]
print(f"Found {len(df_high_wc)} MOFs with CO2 capacity >= 1.5 mmol/g")

# Get list of SMILES for all linkers from high-capacity MOFs
list_smiles = [eval(i) for i in df_high_wc['linkers']]
all_smiles = list(itertools.chain(*list_smiles))
print(f'Total number of SMILES from linkers: {len(all_smiles)}')

# Get unique SMILES
all_smiles_unique = list(pd.Series(all_smiles).unique())
print(f'Number of unique SMILES: {len(all_smiles_unique)}')

# Limit to first 1000 for testing (remove this line for full dataset)
all_smiles_unique = all_smiles_unique[:1000]
print(f'Using {len(all_smiles_unique)} SMILES for processing')

# Output conformers to SDF
print('Generating and outputting conformers to SDF ... ')
conformer_sdf_path = f'data/conformers/conformers_{node_name}.sdf'
conformers_generated = False

if not os.path.isfile(conformer_sdf_path):
    writer = Chem.SDWriter(conformer_sdf_path)
    successful_mols = 0
    
    for smile in tqdm(all_smiles_unique, desc="Generating conformers"):
        try:
            # Create molecule from SMILES
            mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            
            # Sanitize the molecule
            Chem.SanitizeMol(mol)
            
            # Generate conformer
            conformer_id = AllChem.EmbedMolecule(mol)
            
            if conformer_id >= 0:  # Successfully generated conformer
                writer.write(mol)
                successful_mols += 1
            else:
                print(f"Failed to generate conformer for SMILES: {smile}")
                
        except Exception as e:
            print(f"Error processing SMILES {smile}: {e}")
            continue
    
    writer.close()
    conformers_generated = True
    print(f"Successfully generated conformers for {successful_mols} molecules out of {len(all_smiles_unique)}")
else:
    print(f"SDF file already exists: {conformer_sdf_path}")
    conformers_generated = True

# Generate fragment SMILES if SDF was created or already exists
fragment_output_path = f'data/fragments_smi/frag_{node_name}.txt'
if conformers_generated and not os.path.isfile(fragment_output_path):
    print('Generating fragment SMILES ... ')
    
    # Check if the script exists
    script_path = 'utils/prepare_data_from_sdf.py'
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found!")
        print("Please ensure prepare_data_from_sdf.py exists in the utils directory")
    else:
        # Run the fragment generation script
        try:
            subprocess.run([
                'python', script_path,
                '--sdf_path', conformer_sdf_path,
                '--output_path', fragment_output_path,
                '--verbose'
            ], check=True)
            
            if os.path.exists(fragment_output_path):
                print(f"Successfully generated fragment SMILES: {fragment_output_path}")
                # Count lines in output file
                with open(fragment_output_path, 'r') as f:
                    line_count = len(f.readlines())
                print(f"Generated {line_count} fragment SMILES")
            else:
                print(f"Error: Fragment file was not created: {fragment_output_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running fragment generation script: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
elif os.path.isfile(fragment_output_path):
    print(f"Fragment SMILES file already exists: {fragment_output_path}")
    with open(fragment_output_path, 'r') as f:
        line_count = len(f.readlines())
    print(f"Contains {line_count} fragment SMILES")

print(f"\nProcessing complete for node: {node_name}")
print(f"Summary:")
print(f"- Input MOFs: {len(df)}")
print(f"- High capacity MOFs (>=1.5 mmol/g): {len(df_high_wc)}")
print(f"- Unique linker SMILES processed: {len(all_smiles_unique)}")
print(f"- Conformer SDF: {conformer_sdf_path}")
print(f"- Fragment SMILES: {fragment_output_path}")