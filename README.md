# ZrMOF-GenAssemble

Official repository for the automated pipeline "ZrMOF-GenAssemble: A workflow for generative design and assembly of zirconium-based metal-organic frameworks for carbon capture applications".

## Overview

This workflow is an enhanced version based on **GHP-MOFassemble** from the paper:  
**"GHP-MOFassemble: Diffusion modeling, high throughput screening, and molecular dynamics for rational discovery of novel metal-organic frameworks for carbon capture at scale"**  
Authors: Hyun Park, Xiaoli Yan, Ruijie Zhu & Eliu Huerta  
Our framework extends the original work with specific modifications for zirconium-based MOFs and includes additional features for multi-topology assembly and improved property prediction. This workflow integrates diffusion-based generative modeling, automated molecular assembly, and machine learning to enable high-throughput generation and screening of novel zirconium-based MOF structures. The framework combines DiffLinker-generated organic linkers with zirconium-based secondary building units (SBUs) to create diverse Zr-MOF structures with potential applications in carbon capture.

## Key Enhancements Over Original GHP-MOFassemble

1. **Zr-MOF Specialization**: Adapted for zirconium-based MOF generation
2. **Multi-Topology Assembly**: Added Tobacco-based assembly for diverse topologies

## Prerequisites

Required Python packages are listed in `requirements.txt`. Key dependencies include:
- RDKit (for molecular operations)
- PyMatGen (for structure handling)
- DiffLinker (for linker generation)
- CGCNN (for property prediction)
- Tobacco (for MOF assembly)

## Dataset

The workflow begins with high-performing ZrMOF structures from existing databases. Reference structures and their properties are provided in the `data/` directory.

## Workflow

The complete workflow consists of 12 sequential steps:

### Step 1: Fragmenting Linkers
High-performing ZrMOF structures are selected and their linkers are fragmented into molecular building blocks using RDKit's fragmentation algorithms.

### Step 2: Generating Molecular Fragment Conformers
Molecular fragments are converted to 3D conformers and saved in SDF format for subsequent processing.

### Step 3: Sampling New MOF Linkers with DiffLinker
DiffLinker, a diffusion-based generative model, samples novel organic linkers with varying molecular sizes (typically 5-10 atoms).

### Step 4: Converting to All-Atomistic Molecules
Generated linkers are converted to complete molecular structures with identification of dummy atoms for connection points.

### Step 5: Filtering Undesired Linkers
Linkers containing elements unsuitable for Zr-MOFs (S, P, I) are removed to ensure chemical compatibility.

### Step 6: Assembling fcu Zr-MOFs
First-stage assembly creates fcu (face-centered cubic) topology Zr-MOFs using zirconium-based nodes and generated linkers.

### Step 7a: Processing Linkers for Multi-Topology Assembly
Linkers are prepared and formatted for use with the Tobacco assembly tool.

### Step 7b: Assembling Multi-Topology Zr-MOFs with Tobacco
The Tobacco toolkit assembles diverse topologies (cat1, cat2, cat3) using processed linkers and zirconium nodes.

### Step 8: Quality Control
Generated structures are validated and filtered to remove those incompatible with standard computational tools.

### Step 9: Reformatting CIF Files
Crystal structure files are standardized and reformatted for consistency across downstream applications.

### Step 10: Preparing for Property Prediction
Features are extracted and formatted for machine learning model input.

### Step 11: Training Improved CGCNN Model
An enhanced Crystal Graph Convolutional Neural Network is trained to predict CO₂ adsorption properties.

### Step 12: Predicting CO₂ Adsorption Capacity
The trained model predicts CO₂ adsorption capacities for all generated Zr-MOF structures.

## Running the Workflow

Execute the complete workflow with:
```bash
bash generate_Zrmofs.sh
```

Or run individual steps as needed.

## Output

The pipeline generates:
- Novel Zr-MOF structures in multiple topologies
- Predicted CO₂ adsorption capacities
- Quality metrics and validation results
- Trained machine learning models

## Example High-Performing Structures

Predicted high-performing Zr-MOF structures with optimal CO₂ adsorption properties will be available in the `high_performing_ZrMOF_cifs/` directory upon successful pipeline execution.

## Citation

If you use this workflow in your research, please cite:
1. **Original GHP-MOFassemble work**:  
   Park, H., Yan, X., Zhu, R., & Huerta, E. (Year). GHP-MOFassemble: 
2. **Our upcoming publication**:  
   : 

## License

This computational framework is based on GHP-MOFassemble and is released under the CC BY 4.0 License.