echo Step 1 - Fragmenting linkers in high-performing ZrMOF structures into molecular fragments
echo ----------------------------------------------------------------------------------------
python 1_fragmentation.py

echo Step 2 - Generating molecular fragment comformers
echo -------------------------------------------------
python 2_generate_frag_sdf.py

echo Step 3 - Sampling new MOF linkers using DiffLinker
echo --------------------------------------------------
python 3_difflinker.py

echo Step 4 - Converting to all-atomistic molecules and identifying dummy atoms
echo -----------------------------------------------------------------------
python 4_xyz2assemble.py

echo Step 5 - Removing linkers with S, P and I elements
echo --------------------------------------------------
python 5_remove_undesired_linkers.py

echo Step 6 - Assemblying fcu Zr-MOFs
echo ------------------------------
python 6_assemble4ZrMOFs.py

echo Step 7a - Processing linkers for Tobacco
echo ----------------------------------------
python 7a_processlinker_for_tobacco.py

echo Step 7b - Assembling multi-topology ZrMOFs using Tobacco
echo ---------------------------------------------------------
python 7b_tobacco.py

echo Step 8 - Removing structures that cannot be read by pymatgen
echo ------------------------------------------------------------
python 8_remove_invalid_structures.py

echo Step 9 - Reformat cifs
echo ----------------------------------------------------------
python 9_reformat_cif.py

echo Step 10 - Preparing necessary files for making predictions
echo ----------------------------------------------------------
python 10_prep_for_regression.py
