import os
import glob
import shutil
import functools
import subprocess
import pandas as pd
from utils.sascorer import *
from utils.scscorer import *
from multiprocessing import Pool, Manager

linker_base_dir = 'output_for_assembly'

NCPUS = int(0.9*os.cpu_count())

def append_smile(smiles_all, mol, linker_base_dir, n_atoms, sys):
    xyz_path = os.path.join(linker_base_dir, n_atoms, 'xyz_h', sys, mol)
    
    cmd = f'obabel -ixyz "{xyz_path}" -osmi'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    output = result.stdout.strip()
    if output:
        first_line = output.split('\n')[0].strip()
        if first_line:
            parts = first_line.split('\t')
            if parts:
                smi = parts[0].strip()
                if smi:
                    smiles_all.append(f"{smi}\t{xyz_path}\n")

if __name__ == '__main__':
    # remove linkers with S, P and I
    # change to the line below to reproduce paper result
    for n_atom in range(5,11):
        os.makedirs(os.path.join(linker_base_dir,f'n_atoms_{n_atom}','linkers_removed'),exist_ok=True)

    print('Removing linkers with S, P and I ...')
    for n_atoms in sorted(glob.glob(os.path.join(linker_base_dir,'*'))):
        print(f'Now working on n_atoms: {n_atoms}')
        if 'n_atoms' in n_atoms:
            for type_dir in glob.glob(os.path.join(n_atoms,'xyz_*')):
                type_name = os.path.basename(type_dir)
                for sys in os.listdir(type_dir):
                    target_dir = os.path.join(n_atoms, 'linkers_removed', type_name, sys)
                    os.makedirs(target_dir, exist_ok=True)
                    mol_path = os.path.join(type_dir, sys)
                    
                    if os.path.exists(mol_path):
                        for mol in os.listdir(mol_path):
                            file_path = os.path.join(mol_path, mol)
                            
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                            has_undesired = False
                            for line in lines[2:]:
                                if line.strip():
                                    element = line.split()[0]
                                    if element in ['S', 'P', 'I']:
                                        has_undesired = True
                                        break
                            
                            if has_undesired:
                                shutil.move(file_path, os.path.join(target_dir, mol))

    for n_atoms in sorted(os.listdir(linker_base_dir)):
        n_atoms_path = os.path.join(linker_base_dir, n_atoms)
        if os.path.isdir(n_atoms_path) and 'n_atoms' in n_atoms:
            xyz_h_path = os.path.join(n_atoms_path, 'xyz_h')
            
            if not os.path.exists(xyz_h_path):
                continue
                
            for sys in os.listdir(xyz_h_path):
                print(f'Now on {n_atoms} - {sys}')

                # generate smiles
                print('Generating SMILES ...')
                smiles_dir = os.path.join(n_atoms_path, 'smiles')
                os.makedirs(smiles_dir, exist_ok=True)
                
                if '.' not in sys:
                    sys_xyz_path = os.path.join(xyz_h_path, sys)
                    
                    if not os.path.exists(sys_xyz_path) or not os.listdir(sys_xyz_path):
                        print(f"Warning: No files in {sys_xyz_path}. Skipping...")
                        continue

                    manager = Manager()
                    smiles_all = manager.list()

                    with Pool(NCPUS) as p:
                        partial_func = functools.partial(
                            append_smile, 
                            smiles_all,
                            linker_base_dir=linker_base_dir,
                            n_atoms=n_atoms,
                            sys=sys
                        )
                        p.map(partial_func, os.listdir(sys_xyz_path))
                    
                    if not smiles_all:
                        print(f"Warning: No SMILES generated for {sys}. Skipping...")
                        continue 
                    else:
                        smiles_file = os.path.join(smiles_dir, f'{sys}_smi.txt')
                        with open(smiles_file, 'w+') as f:
                            for smi in smiles_all:
                                if smi.strip():  
                                    f.write(smi)
                
                smiles_file = os.path.join(smiles_dir, f'{sys}_smi.txt')
                if not os.path.exists(smiles_file):
                    print(f"Warning: {smiles_file} does not exist. Skipping...")
                    continue

                # calculate SAscore and SCscore
                print('Calculating SAscore and SCscore ...')
                
                # Read smi.txt file and extract SMILES
                with open(smiles_file, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if not lines:
                    print(f"Warning: Empty SMILES file for {sys}. Skipping...")
                    continue
                
                # Parse each line: SMILES + tab + file path
                smiles_list = []
                file_paths = []
                
                for line in lines:
                    # Split by tab character (\t)
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        smi = parts[0].strip()
                        file_path = parts[1].strip()
                        if smi:  # Ensure SMILES is not empty
                            smiles_list.append(smi)
                            file_paths.append(file_path)
                    elif len(parts) == 1:
                        # Only SMILES, no file path
                        smi = parts[0].strip()
                        if smi:
                            smiles_list.append(smi)
                            file_paths.append("")
                
                if not smiles_list:
                    print(f"Warning: No valid SMILES in file for {sys}. Skipping...")
                    continue
                
                print(f"  Processing {len(smiles_list)} molecules...")
                
                # Calculate SAscore
                try:
                    print('  Calculating SAscore...')
                    df_sa = processMols_sa(smiles_list)
                    print(f'    SAscore calculation completed. Shape: {df_sa.shape}')
                    print(f'    SAscore columns: {df_sa.columns.tolist()}')
                except Exception as e:
                    print(f"Error calculating SAscore for {sys}: {e}")
                    import traceback
                    traceback.print_exc()
                    df_sa = pd.DataFrame({'smiles': smiles_list, 'sa_score': [float('nan')]*len(smiles_list)})
                
                # Calculate SCscore
                print('  Calculating SCscore...')
                try:
                    # Import SCScorer directly to ensure proper initialization
                    from utils.scscorer import SCScorer
                    
                    # Create model instance
                    model = SCScorer()
                    
                    # Try different possible weight file locations
                    weights_files = [
                        os.path.join(root_dir, 'utils', 'scscore_1024uint8_model.ckpt-10654.as_numpy.json.gz'),
                        'utils/scscore_1024uint8_model.ckpt-10654.as_numpy.json.gz',
                        'scscore_1024uint8_model.ckpt-10654.as_numpy.json.gz'
                    ]
                    
                    model_loaded = False
                    for wf in weights_files:
                        if os.path.exists(wf):
                            try:
                                model.restore(wf)
                                model_loaded = True
                                print(f'    Model loaded from: {wf}')
                                break
                            except:
                                continue
                    
                    if not model_loaded:
                        raise FileNotFoundError("Could not find SCScorer weight file")
                    
                    # Calculate scores for each SMILES
                    sc_scores = []
                    successful = 0
                    
                    for i, smi in enumerate(smiles_list):
                        try:
                            smi_clean = str(smi).strip()
                            if not smi_clean:
                                sc_scores.append(float('nan'))
                                continue
                                
                            (smi_conv, sco) = model.get_score_from_smi(smi_clean)
                            sc_scores.append(sco)
                            successful += 1
                            
                            # Print first few results
                            if i < 3:
                                smi_short = smi_clean[:40] + "..." if len(smi_clean) > 40 else smi_clean
                                print(f'      {i}: {smi_short} -> SC: {sco:.4f}')
                                
                        except Exception as e:
                            if i < 3:  # Print first few errors only
                                print(f'      Error on SMILES {i}: {str(e)[:50]}')
                            sc_scores.append(float('nan'))
                    
                    # Create DataFrame
                    df_sc = pd.DataFrame({
                        'smiles': smiles_list,
                        'sc_score': sc_scores
                    })
                    
                except Exception as e:
                    print(f"Error calculating SCscore for {sys}: {e}")
                    import traceback
                    traceback.print_exc()
                    df_sc = pd.DataFrame({'smiles': smiles_list, 'sc_score': [float('nan')]*len(smiles_list)})
                
                # Merge two DataFrames
                print('  Merging SA and SC scores...')
                try:
                    # Check DataFrames before merging
                    print(f'    df_sa shape: {df_sa.shape}, columns: {df_sa.columns.tolist()}')
                    print(f'    df_sc shape: {df_sc.shape}, columns: {df_sc.columns.tolist()}')
                    
                    # Ensure both DataFrames have the required columns
                    if 'sa_score' not in df_sa.columns:
                        df_sa['sa_score'] = float('nan')
                    
                    if 'sc_score' not in df_sc.columns:
                        df_sc['sc_score'] = float('nan')
                    
                    # Merge on 'smiles' column
                    df_sa_sc = pd.merge(df_sa, df_sc, on='smiles', how='outer')
                    
                    print(f'    Merged DataFrame shape: {df_sa_sc.shape}')
                    print(f'    Merged columns: {df_sa_sc.columns.tolist()}')
                    
                    # Check for NaN values
                    if 'sa_score' in df_sa_sc.columns:
                        sa_nan = df_sa_sc['sa_score'].isna().sum()
                        print(f'    SA score NaN count: {sa_nan}/{len(df_sa_sc)}')
                    
                    if 'sc_score' in df_sa_sc.columns:
                        sc_nan = df_sa_sc['sc_score'].isna().sum()
                        print(f'    SC score NaN count: {sc_nan}/{len(df_sa_sc)}')
                    
                    # If sc_score column is missing after merge, add it
                    if 'sc_score' not in df_sa_sc.columns:
                        print('    WARNING: sc_score column missing after merge! Adding NaN values.')
                        df_sa_sc['sc_score'] = float('nan')
                    
                except Exception as e:
                    print(f"Error merging scores for {sys}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create combined DataFrame manually
                    df_sa_sc = pd.DataFrame({
                        'smiles': smiles_list,
                        'sa_score': df_sa['sa_score'].tolist() if 'sa_score' in df_sa.columns else [float('nan')]*len(smiles_list),
                        'sc_score': df_sc['sc_score'].tolist() if 'sc_score' in df_sc.columns else [float('nan')]*len(smiles_list)
                    })
                
                # Save scores to CSV
                sc_sa_dir = os.path.join(n_atoms_path, 'sc_sa_score')
                os.makedirs(sc_sa_dir, exist_ok=True)
                score_file = os.path.join(sc_sa_dir, f'{sys}.csv')
                
                try:
                    df_sa_sc.to_csv(score_file, index=False)
                    print(f"  Scores saved to: {score_file}")
                    
                    # Verify file was created
                    if os.path.exists(score_file):
                        # Read back to verify
                        verify_df = pd.read_csv(score_file)
                        print(f'  Verification: file contains {len(verify_df)} rows, {len(verify_df.columns)} columns')
                        print(f'  Columns in saved file: {verify_df.columns.tolist()}')
                    else:
                        print(f'  WARNING: Score file was not created!')
                        
                except Exception as e:
                    print(f"Error saving score file for {sys}: {e}")
                
                # Merge info: combine scores with original file information
                print('  Merging information...')
                info_dir = os.path.join(n_atoms_path, 'info')
                os.makedirs(info_dir, exist_ok=True)
                
                # Create final info DataFrame
                info_data = []
                
                for i, smi in enumerate(smiles_list):
                    # Find scores for this SMILES
                    score_row = df_sa_sc[df_sa_sc['smiles'] == smi]
                    
                    if not score_row.empty and len(score_row) > 0:
                        sa_score = score_row['sa_score'].iloc[0] if 'sa_score' in score_row.columns else float('nan')
                        sc_score = score_row['sc_score'].iloc[0] if 'sc_score' in score_row.columns else float('nan')
                    else:
                        sa_score = float('nan')
                        sc_score = float('nan')
                    
                    # Get filename from file path
                    file_name = ""
                    full_path = ""
                    if i < len(file_paths) and file_paths[i]:
                        full_path = file_paths[i]
                        file_name = os.path.basename(full_path)
                    
                    info_data.append({
                        'file': file_name,
                        'smiles': smi,
                        'sa_score': sa_score,
                        'sc_score': sc_score,
                        'full_path': full_path
                    })
                
                # Create and save info DataFrame
                df_info = pd.DataFrame(info_data)
                info_file = os.path.join(info_dir, f'{sys}_info.csv')
                
                try:
                    df_info.to_csv(info_file, index=False)
                    print(f"  Information saved to: {info_file}")
                    
                    # Print statistics
                    if len(df_info) > 0:
                        valid_sa = df_info['sa_score'].apply(lambda x: not pd.isna(x)).sum()
                        valid_sc = df_info['sc_score'].apply(lambda x: not pd.isna(x)).sum()
                        print(f"  Statistics: {valid_sa}/{len(df_info)} valid SA scores, {valid_sc}/{len(df_info)} valid SC scores")
                    
                    # Show first few rows for verification
                    if len(df_info) > 0:
                        print(f"  Sample data (first 2 rows):")
                        for idx, row in df_info.head(2).iterrows():
                            smi_short = row['smiles'][:40] + "..." if len(row['smiles']) > 40 else row['smiles']
                            print(f"    File: {row['file']}, SA: {row['sa_score']:.4f}, SC: {row['sc_score']:.4f}")
                            print(f"    SMILES: {smi_short}")
                            
                except Exception as e:
                    print(f"Error saving info file for {sys}: {e}")
                
                print(f"  Completed processing for {sys}")
                print("-" * 80)                   

                print('Copying all linkers to target dir')
                for n_atoms_dir in glob.glob(os.path.join(linker_base_dir, 'n_atoms_*')):
                    n_atoms = os.path.basename(n_atoms_dir)
                    
                    xyz_X_zr6_dir = os.path.join(n_atoms_dir, 'xyz_X', 'Zr6')
                    
                    if os.path.exists(xyz_X_zr6_dir):
                        print(f'Processing {n_atoms} - Zr6')
                        
                        target_dir = 'linker_xyz'
                        os.makedirs(target_dir, exist_ok=True)
                        
                        for file in os.listdir(xyz_X_zr6_dir):
                            if file.endswith('.xyz'):
                                base_name = os.path.splitext(file)[0]
                                new_name = f"{base_name}{n_atoms}.xyz"
                                src_file = os.path.join(xyz_X_zr6_dir, file)
                                dst_file = os.path.join(target_dir, new_name)
                                shutil.copy2(src_file, dst_file)
                                print(f"Copied {file} -> {new_name}")
                        
                        print(f"All files from {xyz_X_zr6_dir} have been copied to {target_dir}")