import os
import numpy as np
from ase.io import read
from pymatgen.core.structure import Structure, Lattice
from pymatgen.analysis.local_env import CrystalNN

def process_file(input_xyz, output_cif):
    atoms = read(input_xyz)
    
    try:
        lattice = Lattice(atoms.get_cell())
        if not all(np.linalg.norm(v) > 0.1 for v in lattice.matrix):
            raise ValueError("Invalid lattice parameters")
            
        original_labels = [f"{atom.symbol}{i+1}" for i, atom in enumerate(atoms)]
        structure = Structure(lattice,
                           [atom.symbol for atom in atoms],
                           atoms.get_positions(),
                           coords_are_cartesian=True,
                           to_unit_cell=True,
                           site_properties={"original_label": original_labels})
    except Exception as e:
        max_dim = max(np.ptp(atoms.get_positions()[:,i]) + 10 for i in range(3))
        lattice = Lattice.from_parameters(max_dim, max_dim, max_dim, 90, 90, 90)
        original_labels = [f"{atom.symbol}{i+1}" for i, atom in enumerate(atoms)]
        structure = Structure(lattice,
                           [atom.symbol for atom in atoms],
                           atoms.get_positions(),
                           coords_are_cartesian=True,
                           site_properties={"original_label": original_labels})
    
    at_indices = [i for i, site in enumerate(structure) if site.species_string == "At"]
    
    MIN_BOND_LENGTH = 1.4
    MAX_BOND_LENGTH = 1.8
    
    c_to_label = set()
    for at_idx in at_indices:
        for i, site in enumerate(structure):
            if site.species_string == "C":
                dist = structure[at_idx].distance(site)
                if MIN_BOND_LENGTH <= dist <= MAX_BOND_LENGTH:
                    c_to_label.add(i)
    
    final_labels = []
    for i, site in enumerate(structure):
        original_label = site.properties["original_label"]
        if i in c_to_label:
            final_labels.append(f"X{original_label[1:]}")
        else:
            final_labels.append(original_label)
    
    new_sites = []
    new_labels = []
    for i, site in enumerate(structure):
        if i not in at_indices:
            new_sites.append(site)
            new_labels.append(final_labels[i])
    
    new_structure = Structure.from_sites(new_sites)
    
    bond_info = []
    try:
        cnn = CrystalNN()
        for i, site in enumerate(new_structure):
            neighbors = cnn.get_nn_info(new_structure, i)
            for neighbor in neighbors:
                j = neighbor["site_index"]
                if j > i: 
                    dist = new_structure[i].distance(new_structure[j])
                    bond_info.append((
                        new_labels[i],
                        new_labels[j],
                        f"{dist:.3f}",
                        ".",
                        "S" 
                    ))
    except:
        for i in range(len(new_structure)):
            for j in range(i+1, len(new_structure)):
                dist = new_structure[i].distance(new_structure[j])
                if dist < 2.0: 
                    bond_info.append((
                        new_labels[i],
                        new_labels[j],
                        f"{dist:.3f}",
                        ".",
                        "S"
                    ))
    
    with open(output_cif, 'w') as f:
        f.write("data_\n")
        f.write("_symmetry_space_group_name_H-M 'P 1'\n")
        f.write("_symmetry_int_tables_number 1\n")
        f.write(f"_cell_length_a {new_structure.lattice.a:.6f}\n")
        f.write(f"_cell_length_b {new_structure.lattice.b:.6f}\n")
        f.write(f"_cell_length_c {new_structure.lattice.c:.6f}\n")
        f.write(f"_cell_angle_alpha {new_structure.lattice.alpha:.6f}\n")
        f.write(f"_cell_angle_beta {new_structure.lattice.beta:.6f}\n")
        f.write(f"_cell_angle_gamma {new_structure.lattice.gamma:.6f}\n")
        
        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_site_U_iso_or_equiv\n")
        f.write("_atom_site_adp_type\n")
        f.write("_atom_site_occupancy\n")
        
        for i, site in enumerate(new_structure):
            label = new_labels[i]
            symbol = site.species_string
            x, y, z = site.frac_coords
            f.write(f"{label:<5}    {symbol:<2}    {x:.5f}   {y:.5f}   {z:.5f}   {0.00000:.5f}  Uiso   {1.00:.2f}\n")
        
        f.write("\nloop_\n")
        f.write("_geom_bond_atom_site_label_1\n")
        f.write("_geom_bond_atom_site_label_2\n")
        f.write("_geom_bond_distance\n")
        f.write("_geom_bond_site_symmetry_2\n")
        f.write("_ccdc_geom_bond_type\n")
        
        for bond in bond_info:
            f.write(f"{bond[0]:<5}     {bond[1]:<5}      {bond[2]:<6}  {bond[3]:<1}     {bond[4]:<1}\n")

input_dir = "linker_xyz"
output_dir = "tobacco/edges"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".xyz"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".xyz", ".cif"))
        try:
            process_file(input_path, output_path)
            print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")

print("Processing completed!")