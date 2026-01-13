import os
import io
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymatgen.core as mg
import warnings
import multiprocessing as mproc
from itertools import combinations, product

warnings.filterwarnings("ignore", module="pymatgen")

def read_node_xyz(fpath, dummy_element="At"):
    """Read node XYZ file"""
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    return df

def read_linker_xyz(fpath, dummy_element="At"):
    """Read linker XYZ file and identify anchor positions"""
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    anchor_ids = df[df["el"] == dummy_element].index.tolist()
    
    if len(anchor_ids) != 2:
        raise ValueError(f"Linker should have 2 dummy atoms, found {len(anchor_ids)}")
    
    return df, anchor_ids

def save_xyz_file(df, fpath):
    """Save DataFrame as XYZ file"""
    with io.open(fpath, "w", newline="\n") as wf:
        wf.write(str(len(df)) + "\n\n" + df.to_string(header=None, index=None))

def rotation_matrix_align(vec1, vec2):
    """Generate rotation matrix to align vec1 with vec2"""
    if np.linalg.norm(vec1) < 1e-10 or np.linalg.norm(vec2) < 1e-10:
        return np.eye(3)
    
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        else:
            if abs(a[0]) > 1e-10 or abs(a[1]) > 1e-10:
                v = np.cross(a, [0, 0, 1])
            else:
                v = np.cross(a, [1, 0, 0])
            s = np.linalg.norm(v)
    
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def place_linker_between_points(linker_df, linker_anchors, point1, point2, dummy_element="At"):
    """Place linker between two points in 3D space"""
    linker = linker_df.copy()
    
    linker_anchor1_pos = linker.loc[linker_anchors[0], ["x", "y", "z"]].astype(float).values
    linker_anchor2_pos = linker.loc[linker_anchors[1], ["x", "y", "z"]].astype(float).values
    
    linker_vec = linker_anchor1_pos - linker_anchor2_pos
    target_vec = point1 - point2
    
    linker_center = (linker_anchor1_pos + linker_anchor2_pos) / 2.0
    target_center = (point1 + point2) / 2.0
    
    if np.linalg.norm(linker_vec) > 1e-10 and np.linalg.norm(target_vec) > 1e-10:
        rot_mat = rotation_matrix_align(linker_vec, target_vec)
        
        linker_coords = linker[["x", "y", "z"]].astype(float).values
        linker_coords_centered = linker_coords - linker_center
        linker_coords_rotated = np.dot(linker_coords_centered, rot_mat.T)
        linker_coords_final = linker_coords_rotated + target_center
        
        linker_anchor1_rotated = np.dot(linker_anchor1_pos - linker_center, rot_mat.T) + target_center
        linker_anchor2_rotated = np.dot(linker_anchor2_pos - linker_center, rot_mat.T) + target_center
        
        adjustment1 = point1 - linker_anchor1_rotated
        adjustment2 = point2 - linker_anchor2_rotated
        final_adjustment = (adjustment1 + adjustment2) / 2.0
        
        linker_coords_final += final_adjustment
        linker.loc[:, ["x", "y", "z"]] = linker_coords_final
    
    linker_atoms = []
    for idx, row in linker.iterrows():
        if row["el"] != dummy_element:
            linker_atoms.append({
                "el": row["el"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"])
            })
    
    if len(linker_atoms) == 0:
        raise ValueError("No non-dummy atoms found after linker placement")
    
    return linker_atoms

def place_node_at_position(node_df, position, dummy_element="At"):
    """Place node at specified position"""
    node_coords = node_df[["x", "y", "z"]].astype(float).values
    node_center = np.mean(node_coords, axis=0)
    translation = position - node_center
    
    node_atoms = []
    for idx, row in node_df.iterrows():
        if row["el"] != dummy_element:
            atom_pos = row[["x", "y", "z"]].astype(float).values + translation
            node_atoms.append({
                "el": row["el"],
                "x": float(atom_pos[0]),
                "y": float(atom_pos[1]),
                "z": float(atom_pos[2])
            })
    
    return node_atoms

def apply_symmetry_operation(node_idx, sym_code, node_positions_frac, lattice):
    """Apply symmetry operation to node position"""
    original_pos_frac = np.array(node_positions_frac[node_idx], dtype=float)
    
    if sym_code == '.':
        return np.dot(original_pos_frac, lattice)
    
    parts = sym_code.split('_')
    if len(parts) != 2:
        return np.dot(original_pos_frac, lattice)
    
    trans_code = parts[1]
    if len(trans_code) != 3:
        return np.dot(original_pos_frac, lattice)
    
    result_pos_frac = original_pos_frac.copy()
    
    for i, code in enumerate(trans_code):
        if code == '4':
            result_pos_frac[i] -= 1.0
        elif code == '6':
            result_pos_frac[i] += 1.0
    
    return np.dot(result_pos_frac, lattice)

def calculate_lattice_constant_single(node_df, linker_df, dummy_element="At"):
    """Calculate lattice constant for single linker structures"""
    node_coords = node_df[node_df["el"] != dummy_element][["x", "y", "z"]].astype(float).values
    if len(node_coords) == 0:
        raise ValueError("No non-dummy atoms found in node file")
    
    node_min = np.min(node_coords, axis=0)
    node_max = np.max(node_coords, axis=0)
    node_diameter = np.max(node_max - node_min)
    
    linker_coords = linker_df[linker_df["el"] != dummy_element][["x", "y", "z"]].astype(float).values
    if len(linker_coords) == 0:
        raise ValueError("No non-dummy atoms found in linker file")
    
    linker_min = np.min(linker_coords, axis=0)
    linker_max = np.max(linker_coords, axis=0)
    linker_length = np.max(linker_max - linker_min)
    
    dummy_indices = linker_df[linker_df["el"] == dummy_element].index.tolist()
    if len(dummy_indices) >= 2:
        dummy_pos1 = linker_df.loc[dummy_indices[0], ["x", "y", "z"]].astype(float).values
        dummy_pos2 = linker_df.loc[dummy_indices[1], ["x", "y", "z"]].astype(float).values
        linker_effective_length = np.linalg.norm(dummy_pos2 - dummy_pos1)
    else:
        linker_effective_length = linker_length
    
    node_radius = node_diameter / 2.0
    node_connector_distance = node_radius * 0.8
    
    inter_node_distance = 2 * node_connector_distance + linker_effective_length
    lattice_constant = inter_node_distance / 0.7071
    lattice_constant = lattice_constant * 1.15
    
    return round(lattice_constant, 2)

def calculate_lattice_constant_dual(node_df, linker_dfs, dummy_element="At"):
    """Calculate lattice constant for dual linker structures"""
    node_coords = node_df[node_df["el"] != dummy_element][["x", "y", "z"]].astype(float).values
    if len(node_coords) == 0:
        raise ValueError("No non-dummy atoms found in node file")
    
    node_min = np.min(node_coords, axis=0)
    node_max = np.max(node_coords, axis=0)
    node_diameter = np.max(node_max - node_min)
    
    # Calculate maximum dimensions from all linkers
    max_linker_length = 0
    max_effective_length = 0
    
    for linker_df in linker_dfs:
        linker_coords = linker_df[linker_df["el"] != dummy_element][["x", "y", "z"]].astype(float).values
        if len(linker_coords) == 0:
            continue
        
        linker_min = np.min(linker_coords, axis=0)
        linker_max = np.max(linker_coords, axis=0)
        linker_length = np.max(linker_max - linker_min)
        max_linker_length = max(max_linker_length, linker_length)
        
        dummy_indices = linker_df[linker_df["el"] == dummy_element].index.tolist()
        if len(dummy_indices) >= 2:
            dummy_pos1 = linker_df.loc[dummy_indices[0], ["x", "y", "z"]].astype(float).values
            dummy_pos2 = linker_df.loc[dummy_indices[1], ["x", "y", "z"]].astype(float).values
            linker_effective_length = np.linalg.norm(dummy_pos2 - dummy_pos1)
            max_effective_length = max(max_effective_length, linker_effective_length)
        else:
            max_effective_length = max(max_effective_length, linker_length)
    
    if max_linker_length == 0:
        raise ValueError("No valid linker dimensions found")
    
    node_radius = node_diameter / 2.0
    node_connector_distance = node_radius * 0.8
    
    inter_node_distance = 2 * node_connector_distance + max_effective_length
    lattice_constant = inter_node_distance / 0.7071
    lattice_constant = lattice_constant * 1.15
    
    return round(lattice_constant, 2)

def build_fcu_structure_single(node_df, linker_df, linker_anchors, lattice_constant=None, dummy_element="At"):
    """Build FCU structure with single linker type"""
    all_atoms = []
    
    if lattice_constant is None:
        lattice_constant = calculate_lattice_constant_single(node_df, linker_df, dummy_element)
    
    lattice = np.array([[lattice_constant, 0, 0],
                        [0, lattice_constant, 0],
                        [0, 0, lattice_constant]])
    
    node_positions_frac = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ]
    
    node_positions = [np.dot(pos_frac, lattice) for pos_frac in node_positions_frac]
    
    for pos in node_positions:
        node_atoms = place_node_at_position(node_df, pos, dummy_element)
        all_atoms.extend(node_atoms)
    
    connections = [
        (0, 1, '.'), (0, 1, '1_545'), (0, 1, '1_544'), (0, 1, '1_554'),
        (0, 2, '1_455'), (0, 2, '1_554'), (0, 2, '1_454'), (0, 2, '.'),
        (0, 3, '.'), (0, 3, '1_545'), (0, 3, '1_455'), (0, 3, '1_445'),
        (1, 2, '1_565'), (1, 2, '.'), (1, 2, '1_465'), (1, 2, '1_455'),
        (1, 3, '1_456'), (1, 3, '1_556'), (1, 3, '.'), (1, 3, '1_455'),
        (2, 3, '1_556'), (2, 3, '1_546'), (2, 3, '1_545'), (2, 3, '.'),
    ]
    
    successful_placements = 0
    
    for idx1, idx2, sym_code in connections:
        pos1_cart = node_positions[idx1]
        pos2_cart_sym = apply_symmetry_operation(idx2, sym_code, node_positions_frac, lattice)
        
        try:
            linker_atoms = place_linker_between_points(
                linker_df.copy(), 
                linker_anchors,
                pos1_cart,
                pos2_cart_sym,
                dummy_element
            )
            
            if len(linker_atoms) > 0:
                all_atoms.extend(linker_atoms)
                successful_placements += 1
        
        except Exception:
            continue
    
    return all_atoms, lattice, successful_placements

def build_fcu_structure_dual(node_df, linker1_df, linker1_anchors, linker2_df, linker2_anchors, 
                            linker1_ratio=0.5, lattice_constant=None, dummy_element="At"):
    """Build FCU structure with two different linkers"""
    all_atoms = []
    
    if lattice_constant is None:
        lattice_constant = calculate_lattice_constant_dual(node_df, [linker1_df, linker2_df], dummy_element)
    
    lattice = np.array([[lattice_constant, 0, 0],
                        [0, lattice_constant, 0],
                        [0, 0, lattice_constant]])
    
    node_positions_frac = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ]
    
    node_positions = [np.dot(pos_frac, lattice) for pos_frac in node_positions_frac]
    
    for pos in node_positions:
        node_atoms = place_node_at_position(node_df, pos, dummy_element)
        all_atoms.extend(node_atoms)
    
    connections = [
        (0, 1, '.'), (0, 1, '1_545'), (0, 1, '1_544'), (0, 1, '1_554'),
        (0, 2, '1_455'), (0, 2, '1_554'), (0, 2, '1_454'), (0, 2, '.'),
        (0, 3, '.'), (0, 3, '1_545'), (0, 3, '1_455'), (0, 3, '1_445'),
        (1, 2, '1_565'), (1, 2, '.'), (1, 2, '1_465'), (1, 2, '1_455'),
        (1, 3, '1_456'), (1, 3, '1_556'), (1, 3, '.'), (1, 3, '1_455'),
        (2, 3, '1_556'), (2, 3, '1_546'), (2, 3, '1_545'), (2, 3, '.'),
    ]
    
    successful_placements = 0
    linker1_count = 0
    linker2_count = 0
    
    for idx, (idx1, idx2, sym_code) in enumerate(connections):
        pos1_cart = node_positions[idx1]
        pos2_cart_sym = apply_symmetry_operation(idx2, sym_code, node_positions_frac, lattice)
        
        # Determine which linker to use based on ratio
        if idx / len(connections) < linker1_ratio:
            current_linker_df = linker1_df
            current_anchors = linker1_anchors
            linker_type = 1
        else:
            current_linker_df = linker2_df
            current_anchors = linker2_anchors
            linker_type = 2
        
        try:
            linker_atoms = place_linker_between_points(
                current_linker_df.copy(), 
                current_anchors,
                pos1_cart,
                pos2_cart_sym,
                dummy_element
            )
            
            if len(linker_atoms) > 0:
                all_atoms.extend(linker_atoms)
                successful_placements += 1
                if linker_type == 1:
                    linker1_count += 1
                else:
                    linker2_count += 1
        
        except Exception:
            # Try the other linker if placement fails
            try:
                other_linker_df = linker2_df if linker_type == 1 else linker1_df
                other_anchors = linker2_anchors if linker_type == 1 else linker1_anchors
                
                linker_atoms = place_linker_between_points(
                    other_linker_df.copy(), 
                    other_anchors,
                    pos1_cart,
                    pos2_cart_sym,
                    dummy_element
                )
                
                if len(linker_atoms) > 0:
                    all_atoms.extend(linker_atoms)
                    successful_placements += 1
                    if linker_type == 1:
                        linker2_count += 1
                    else:
                        linker1_count += 1
            
            except Exception:
                continue
    
    return all_atoms, lattice, successful_placements, linker1_count, linker2_count

def assemble_fcu_structure_single(node_path, linker_path, output_dir, dummy_element="At", lattice_constant=None):
    """Assemble FCU structure with single linker"""
    os.makedirs(output_dir, exist_ok=True)
    
    node_df = read_node_xyz(node_path, dummy_element)
    linker_df, anchors = read_linker_xyz(linker_path, dummy_element)
    
    all_atoms, lattice, successful_placements = build_fcu_structure_single(
        node_df, linker_df, anchors, lattice_constant, dummy_element
    )
    
    final_df = pd.DataFrame(all_atoms)
    
    node_name = os.path.basename(node_path).replace('.xyz', '')
    linker_name = os.path.basename(linker_path).replace('.xyz', '')
    
    base_name = f"{node_name}_{linker_name}"
    xyz_path = os.path.join(output_dir, f"{base_name}.xyz")
    cif_path = os.path.join(output_dir, f"{base_name}.cif")
    
    save_xyz_file(final_df, xyz_path)
    
    coords = final_df[["x", "y", "z"]].values
    elements = final_df["el"].tolist()
    
    mol = mg.Molecule(elements, coords)
    structure = mg.Structure(
        lattice,
        mol.species,
        mol.cart_coords,
        coords_are_cartesian=True
    )
    
    structure.to(filename=cif_path, fmt="cif")
    
    return cif_path, len(all_atoms), successful_placements

def assemble_fcu_structure_dual(node_path, linker1_path, linker2_path, output_dir, 
                              dummy_element="At", lattice_constant=None, linker1_ratio=0.5):
    """Assemble FCU structure with two different linkers"""
    os.makedirs(output_dir, exist_ok=True)
    
    node_df = read_node_xyz(node_path, dummy_element)
    linker1_df, anchors1 = read_linker_xyz(linker1_path, dummy_element)
    linker2_df, anchors2 = read_linker_xyz(linker2_path, dummy_element)
    
    all_atoms, lattice, successful_placements, linker1_count, linker2_count = build_fcu_structure_dual(
        node_df, linker1_df, anchors1, linker2_df, anchors2, linker1_ratio, lattice_constant, dummy_element
    )
    
    final_df = pd.DataFrame(all_atoms)
    
    node_name = os.path.basename(node_path).replace('.xyz', '')
    linker1_name = os.path.basename(linker1_path).replace('.xyz', '')
    linker2_name = os.path.basename(linker2_path).replace('.xyz', '')
    
    base_name = f"{node_name}_{linker1_name}_{linker2_name}"
    xyz_path = os.path.join(output_dir, f"{base_name}.xyz")
    cif_path = os.path.join(output_dir, f"{base_name}.cif")
    
    save_xyz_file(final_df, xyz_path)
    
    coords = final_df[["x", "y", "z"]].values
    elements = final_df["el"].tolist()
    
    mol = mg.Molecule(elements, coords)
    structure = mg.Structure(
        lattice,
        mol.species,
        mol.cart_coords,
        coords_are_cartesian=True
    )
    
    structure.to(filename=cif_path, fmt="cif")
    
    return cif_path, len(all_atoms), successful_placements, linker1_count, linker2_count

def assemble_single_wrapper(args):
    """Wrapper for single linker multiprocessing"""
    node_path, linker_path, output_dir, dummy_element, lattice_constant = args
    try:
        return assemble_fcu_structure_single(
            node_path, linker_path, output_dir, dummy_element, lattice_constant
        )
    except Exception as e:
        print(f"Error assembling {linker_path}: {str(e)}")
        return None

def assemble_dual_wrapper(args):
    """Wrapper for dual linker multiprocessing"""
    node_path, linker1_path, linker2_path, output_dir, dummy_element, lattice_constant, linker1_ratio = args
    try:
        return assemble_fcu_structure_dual(
            node_path, linker1_path, linker2_path, output_dir, dummy_element, lattice_constant, linker1_ratio
        )
    except Exception as e:
        print(f"Error assembling {linker1_path} + {linker2_path}: {str(e)}")
        return None

def generate_single_linker_tasks(node_path, linker_dir, output_dir, dummy_element="At"):
    """Generate tasks for single linker structures"""
    all_linkers = [os.path.join(linker_dir, f) for f in os.listdir(linker_dir) 
                   if f.endswith('.xyz')]
    all_linkers.sort()
    
    task_args = []
    for linker_path in all_linkers:
        task_args.append((
            node_path,
            linker_path,
            output_dir,
            dummy_element,
            None
        ))
    
    return all_linkers, task_args

def generate_dual_linker_tasks(node_path, linker_dir, output_dir, combination_mode="all_pairs", 
                              dummy_element="At", linker1_ratio=0.5):
    """Generate tasks for dual linker structures"""
    all_linkers = [os.path.join(linker_dir, f) for f in os.listdir(linker_dir) 
                   if f.endswith('.xyz')]
    all_linkers.sort()
    
    if combination_mode == "all_pairs":
        # Generate all unique pairs
        combinations_list = list(combinations(all_linkers, 2))
    elif combination_mode == "self_pairs":
        # Include self-pairs (same linker)
        combinations_list = []
        for i in range(len(all_linkers)):
            for j in range(i, len(all_linkers)):
                combinations_list.append((all_linkers[i], all_linkers[j]))
    elif combination_mode == "all":
        # All possible combinations including self and repeats
        combinations_list = list(product(all_linkers, repeat=2))
    else:
        raise ValueError(f"Unknown combination mode: {combination_mode}")
    
    task_args = []
    for linker1_path, linker2_path in combinations_list:
        task_args.append((
            node_path,
            linker1_path,
            linker2_path,
            output_dir,
            dummy_element,
            None,
            linker1_ratio
        ))
    
    return combinations_list, task_args

def run_single_linker_generation(node_path, linker_dir, output_dir, NCPUS):
    """Run single linker structure generation"""
    print("\n" + "="*60)
    print("STEP 1: Generating Single Linker MOFs")
    print("="*60)
    
    if not os.path.exists(node_path):
        raise FileNotFoundError(f"Node file not found: {node_path}")
    
    if not os.path.exists(linker_dir):
        raise FileNotFoundError(f"Linker directory not found: {linker_dir}")
    
    all_linkers, task_args = generate_single_linker_tasks(
        node_path, linker_dir, output_dir, "At"
    )
    
    if len(task_args) == 0:
        print(f"No linker files found in {linker_dir}")
        return 0, []
    
    print(f"Found {len(all_linkers)} linkers")
    print(f"Will generate {len(task_args)} MOF structures")
    print(f"Output directory: {output_dir}")
    print(f"Using {min(NCPUS, len(task_args))} CPU cores")
    
    successful_count = 0
    failed_structures = []
    
    with mproc.Pool(processes=min(NCPUS, len(task_args))) as pool:
        with tqdm(total=len(task_args), desc="Generating Single Linker MOFs") as pbar:
            for i, result in enumerate(pool.imap_unordered(assemble_single_wrapper, task_args)):
                pbar.update(1)
                if result is None:
                    failed_structures.append(all_linkers[i])
                else:
                    successful_count += 1
    
    print(f"\nSingle Linker Generation completed!")
    print(f"Successful: {successful_count}/{len(task_args)}")
    
    if failed_structures:
        print(f"Failed: {len(failed_structures)} structures")
        for linker in failed_structures[:10]:
            linker_name = os.path.basename(linker)
            print(f"  - {linker_name}")
        if len(failed_structures) > 10:
            print(f"  ... and {len(failed_structures)-10} more")
    
    # Save statistics
    stats_file = os.path.join(output_dir, "generation_stats_single.txt")
    with open(stats_file, 'w') as f:
        f.write("FCU Structure Generation Statistics (Single Linker)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Node file: {node_path}\n")
        f.write(f"Linker directory: {linker_dir}\n")
        f.write(f"Total linkers: {len(all_linkers)}\n")
        f.write(f"Node positions: 4\n")
        f.write(f"Total connections: 24\n")
        f.write(f"Attempted: {len(task_args)}\n")
        f.write(f"Successful: {successful_count}\n")
        f.write(f"Success rate: {successful_count/len(task_args)*100:.1f}%\n")
        f.write(f"Lattice constant: Auto-calculated with 1.15 safety factor\n")
        f.write(f"Output directory: {os.path.abspath(output_dir)}\n")
        f.write(f"Completion time: {pd.Timestamp.now()}\n")
        
        if failed_structures:
            f.write(f"\nFailed structures:\n")
            for linker in failed_structures:
                linker_name = os.path.basename(linker)
                f.write(f"  {linker_name}\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    
    return successful_count, failed_structures

def run_dual_linker_generation(node_path, linker_dir, output_dir, NCPUS, combination_mode="all_pairs"):
    """Run dual linker structure generation"""
    print("\n" + "="*60)
    print("STEP 2: Generating Dual Linker MOFs")
    print("="*60)
    
    if not os.path.exists(node_path):
        raise FileNotFoundError(f"Node file not found: {node_path}")
    
    if not os.path.exists(linker_dir):
        raise FileNotFoundError(f"Linker directory not found: {linker_dir}")
    
    combinations_list, task_args = generate_dual_linker_tasks(
        node_path, linker_dir, output_dir, combination_mode, "At", 0.5
    )
    
    if len(task_args) == 0:
        print(f"No linker combinations found in {linker_dir}")
        return 0, []
    
    print(f"Found {len(combinations_list)} linker combinations")
    print(f"Will generate {len(task_args)} MOF structures")
    print(f"Output directory: {output_dir}")
    print(f"Using {min(NCPUS, len(task_args))} CPU cores")
    print(f"Combination mode: {combination_mode}")
    print(f"Linker ratio: 50/50")
    
    successful_count = 0
    failed_structures = []
    
    with mproc.Pool(processes=min(NCPUS, len(task_args))) as pool:
        with tqdm(total=len(task_args), desc="Generating Dual Linker MOFs") as pbar:
            for i, result in enumerate(pool.imap_unordered(assemble_dual_wrapper, task_args)):
                pbar.update(1)
                if result is None:
                    linker1, linker2 = combinations_list[i]
                    failed_structures.append((linker1, linker2))
                else:
                    successful_count += 1
    
    print(f"\nDual Linker Generation completed!")
    print(f"Successful: {successful_count}/{len(task_args)}")
    
    if failed_structures:
        print(f"Failed: {len(failed_structures)} structures")
        for i, (linker1, linker2) in enumerate(failed_structures[:10]):
            linker1_name = os.path.basename(linker1)
            linker2_name = os.path.basename(linker2)
            print(f"  - {linker1_name} + {linker2_name}")
        if len(failed_structures) > 10:
            print(f"  ... and {len(failed_structures)-10} more")
    
    # Save statistics
    stats_file = os.path.join(output_dir, "generation_stats_dual.txt")
    with open(stats_file, 'w') as f:
        f.write("FCU Structure Generation Statistics (Dual Linker)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Node file: {node_path}\n")
        f.write(f"Linker directory: {linker_dir}\n")
        f.write(f"Total linker combinations: {len(combinations_list)}\n")
        f.write(f"Combination mode: {combination_mode}\n")
        f.write(f"Node positions: 4\n")
        f.write(f"Total connections: 24\n")
        f.write(f"Linker ratio: 50/50\n")
        f.write(f"Attempted: {len(task_args)}\n")
        f.write(f"Successful: {successful_count}\n")
        f.write(f"Success rate: {successful_count/len(task_args)*100:.1f}%\n")
        f.write(f"Lattice constant: Auto-calculated with 1.15 safety factor\n")
        f.write(f"Output directory: {os.path.abspath(output_dir)}\n")
        f.write(f"Completion time: {pd.Timestamp.now()}\n")
        
        if failed_structures:
            f.write(f"\nFailed structures:\n")
            for linker1, linker2 in failed_structures:
                linker1_name = os.path.basename(linker1)
                linker2_name = os.path.basename(linker2)
                f.write(f"  {linker1_name} + {linker2_name}\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    
    return successful_count, failed_structures

def main():
    """Main function to generate both single and dual linker MOFs"""
    NCPUS = max(1, int(0.9 * os.cpu_count()))
    
    node_name = "Zr12"
    node_path = f"node_xyz/{node_name}.xyz"
    
    linker_dir = "linker_xyz"
    
    # Step 1: Generate single linker MOFs
    output_single_dir = f"newMOFs/{node_name}"
    single_success, single_failed = run_single_linker_generation(
        node_path, linker_dir, output_single_dir, NCPUS
    )
    
    # Step 2: Generate dual linker MOFs
    output_dual_dir = f"newMOFs/{node_name}_dual_linker"
    dual_success, dual_failed = run_dual_linker_generation(
        node_path, linker_dir, output_dual_dir, NCPUS, combination_mode="all_pairs"
    )
    
    # Final summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Single Linker MOFs:")
    print(f"  - Output directory: {os.path.abspath(output_single_dir)}")
    print(f"  - Successful: {single_success}")
    print(f"  - Failed: {len(single_failed)}")
    
    print(f"\nDual Linker MOFs:")
    print(f"  - Output directory: {os.path.abspath(output_dual_dir)}")
    print(f"  - Successful: {dual_success}")
    print(f"  - Failed: {len(dual_failed)}")
    
    print(f"\nTotal MOFs generated: {single_success + dual_success}")
    print("="*60)

if __name__ == "__main__":
    main()