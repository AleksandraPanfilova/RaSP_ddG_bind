import glob
import os
import random
import sys
import subprocess

import numpy as np
import pandas as pd
import torch
#import vaex
from Bio.PDB.Polypeptide import index_to_one, three_to_one
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from torch.utils.data import DataLoader, Dataset

import speedtest

from rasp_model import (
    CavityModel,
    DownstreamModel,
    ResidueEnvironment,
    ResidueEnvironmentsDataset,
)
from helpers import (
    compute_pdb_combo_corrs,
    ds_pred,
    ds_train_val,
    fermi_transform,
    get_ddg_dataloader,
    init_lin_weights,
    inverse_fermi_transform,
    populate_dfs_with_resenvs,
    remove_disulfides,
    train_loop,
    train_val_split_cavity,
    train_val_split_ds,
    bootstrap_gnomad_clinvar,
)
from visualization import (
    hist_plot_all,
    homology_plot,
    learning_curve_ds_with_errorbars,
    loss_analysis,
    plot_gnomad_clinvar,
    scatter_plots,
)

# Set fixed seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# Cavity constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
BATCH_SIZE_CAVITY = 100
SHUFFLE_PDBS_CAVITY = True
EPOCHS_CAVITY = 20
PATIENCE_CUTOFF = 6

# Downstream constants
BATCH_SIZE_DS = 40
NUM_ENSEMBLE = 10

# Init
parser = PDBParser()
io = PDBIO()

def main():
    # Pre-process all protein structures
    print(f"Pre-processing PDBs ...")
    pdb_dir = f"{os.path.dirname(sys.path[0])}/data/test/Skempi/structure/"
    
    ## Split into chains
    #from Bio.PDB import PDBParser
    #from Bio.PDB.PDBIO import PDBIO
    #pdb_filenames = glob.glob(f"{pdb_dir}/raw/*.pdb")
    #for pdb_filename in pdb_filenames:
    #    pdb_id = pdb_filename.split("/")[-1].split(".")[0]
    #    structure = parser.get_structure(pdb_id, pdb_filename)
    #    pdb_chains = structure.get_chains()
    #    for chain in pdb_chains:
    #        io.set_structure(chain)
    #        io.save(f"{pdb_dir}/raw/{pdb_id}_{chain.get_id()}.pdb")
   
    # OBS: 1KBH.pdb is for some reason extremely slow to clean --> We have created structure/raw directory without this (containsly only multi mutants)
    #subprocess.call(
    #    [
    #        f"{os.path.dirname(sys.path[0])}/src/pdb_parser_scripts/parse_pdbs_pred.sh",
    #        str(pdb_dir),
    #    ]
    #)
    #print("Pre-processing finished.")

    # Load structure data
    pdb_filenames_cleaned = sorted(
            glob.glob(
                f"{os.path.dirname(sys.path[0])}/data/test/Skempi/structure/cleaned/*.pdb"
            )
        )
  
    # Create fake Rosetta data
    df_list = []
    for pdb_filename in pdb_filenames_cleaned:
        pdb_id = "_".join(pdb_filename.split("/")[-1].split(".")[0].split("_")[:-1])
        structure = parser.get_structure(pdb_id, pdb_filename)
        first_model = structure.get_list()[0]
        first_model.child_list = sorted(first_model.child_list) # Sort chains alphabetically
        
        for chain_model in first_model.child_list:
            variant_list = []
            for i, res in enumerate(chain_model):
                variant_base = three_to_one(res.resname) + str(res._id[1])
                for j in range(20):
                    variant = variant_base + index_to_one(j)
                    df_list.append([pdb_id, 
                                    chain_model._id,
                                    variant
                                    ]
                                   )

    # Convert to df and save
    df = pd.DataFrame(df_list, columns=["pdbid", "chainid", "variant"])
    df["score_rosetta"] = 999.0
    df.to_csv("../data/test/Skempi/ddG_Rosetta/ddg.csv", index=False)

    # Load Rosetta ddGs
    df_rosetta = pd.read_csv(
                        f"{os.path.dirname(sys.path[0])}/data/test/Skempi/ddG_Rosetta/ddg.csv"
                                )
    
    # Create temporary residue environment datasets to more easily match ddG data
    pdb_filenames_parsed = sorted(glob.glob(f"{os.path.dirname(sys.path[0])}/data/test/Skempi/structure/parsed/*coord*"))
    dataset_key = "Skempi"
    dataset_structure = ResidueEnvironmentsDataset(pdb_filenames_parsed, dataset_key=dataset_key)
    resenv_dataset = {}
    for resenv in dataset_structure:
        key = (
            f"{resenv.pdb_id}{resenv.chain_id}_{resenv.pdb_residue_number}"
            f"{index_to_one(resenv.restype_index)}"
        )
        resenv_dataset[key] = resenv

    # Populate Rosetta dataframes with wt ResidueEnvironment objects and wt and mt restype indices
    n_rosetta_start = len(df_rosetta)
    populate_dfs_with_resenvs(df_rosetta, resenv_dataset)
    print(
        f"{n_rosetta_start-len(df_rosetta)} data points dropped when (inner) matching Rosetta and structure in: {dataset_key} data set."
    )
    df_total = df_rosetta

    # Initialize models
    cavity_model_net = CavityModel(get_latent=True).to(DEVICE)
    best_cavity_model_path = (
        open(
            f"{os.path.dirname(sys.path[0])}/output/cavity_models/best_model_path.txt",
            "r",
        )
        .read()
        .strip()
    )
    cavity_model_net.load_state_dict(
        torch.load(f"{os.path.dirname(sys.path[0])}/output/cavity_models/{best_cavity_model_path}", 
                   map_location=torch.device(DEVICE)
        )
    )
    cavity_model_net.eval()
    ds_model_net = DownstreamModel().to(DEVICE)

    # Make test set predictions
    df_ml = ds_pred(
        cavity_model_net, ds_model_net, df_total, dataset_key, NUM_ENSEMBLE, DEVICE
    )

    # Compute dddGs
    df_ml["score_ml_ddg_bind"] = np.nan
    for pdbid in df_ml["pdbid"].unique():
        if "_" not in pdbid:
            df_ml_pdb = df_ml[df_ml["pdbid"] == pdbid]
            
            for chainid in df_ml_pdb["chainid"].unique():
                df_ml_pdb_chain = df_ml_pdb[df_ml_pdb["chainid"] == chainid].reset_index()
                df_ml_pdb_chain_single = df_ml[df_ml["pdbid"] == f"{pdbid}_{chainid}"].reset_index()
                df_ml.loc[((df_ml["pdbid"]==pdbid) & (df_ml["chainid"] == chainid)),"score_ml_ddg_bind"] = (df_ml_pdb_chain["score_ml"] - df_ml_pdb_chain_single["score_ml"]).values

    # Save
    df_ml.to_csv(f"{os.path.dirname(sys.path[0])}/output/{dataset_key}/df_ml.csv", index=False)
    print(f"Finished making predictions")

    # TO DO: Match with experimental dddGs (single-point mutations only)
    # Do this offline locally?

if __name__ == "__main__":
    main()
