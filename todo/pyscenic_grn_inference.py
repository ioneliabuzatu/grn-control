import os, glob, re, pickle
from functools import partial
from collections import OrderedDict
import operator as op
from cytoolz import compose
import pandas as pd
import seaborn as sns
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyscenic.export import export2loom, add_scenic_metadata
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons
from pyscenic.aucell import aucell

#################### setup files and variables ############################################################
wd = "pySCENIC_data/" #working directory
RESOURCES_FOLDERNAME = wd + "resources/"
AUXILLIARIES_FOLDERNAME = wd + "auxilliaries/"
RESULTS_FOLDERNAME = wd + "results/"
FIGURES_FOLDERNAME = wd + "figures/"
BASE_URL = "http://motifcollections.aertslab.org/v9/logos/"
COLUMN_NAME_LOGO = "MotifLogo"
COLUMN_NAME_MOTIF_ID = "MotifID"
COLUMN_NAME_TARGETS = "TargetGenes"
HUMAN_TFS_FNAME = os.path.join(AUXILLIARIES_FOLDERNAME, 'lambert2018.txt')
RANKING_DBS_FNAMES = list(map(lambda fn: os.path.join(AUXILLIARIES_FOLDERNAME, fn),
                              ['hg19-500bp-upstream10species.mc9nr.feather',
                                'hg19-tss-centered-5kb10species.mc9nr.feather',
                                'hg19-tss-centered-10kb10species.mc9nr.feather']))
MOTIF_ANNOTATIONS_FNAME = os.path.join(AUXILLIARIES_FOLDERNAME, 'motifsv9- nr.hgnc-m0.001-o0.0.tbl')
DATASET_ID = "Experiment_name" # for example GEO ID
TCGA_CODE = 'CANCERTYPE' # TCGA cancer code.
CELL_ANNOTATIONS_FNAME = os.path.join(RESOURCES_FOLDERNAME, " cell.annotations.csv")
SAMPLE_METADATA_FNAME = os.path.join(RESOURCES_FOLDERNAME, "meta_data.xlsx")
EXP_MTX_TPM_FNAME = os.path.join(RESOURCES_FOLDERNAME, 'Expression_tpm.csv')
EXP_MTX_COUNTS_FNAME = os.path.join(RESOURCES_FOLDERNAME, 'Expression_counts.csv')
# Output files and folders'
METADATA_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.metadata.csv'.format(DATASET_ID))
EXP_MTX_QC_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.qc.tpm.csv'.format(DATASET_ID))
ADJACENCIES_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.adjacencies.tsv'.format(DATASET_ID))
MOTIFS_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.motifs.csv'.format(DATASET_ID))
REGULONS_DAT_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.regulons.dat'.format(DATASET_ID))
AUCELL_MTX_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.auc.csv'.format(DATASET_ID))
BIN_MTX_FNAME = os.path.join(RESULTS_FOLDERNAME , '{}.bin.csv'.format(DATASET_ID))
THR_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.thresholds.csv'.format(DATASET_ID))
ANNDATA_FNAME = os.path.join(RESULTS_FOLDERNAME , '{}.h5ad'.format(DATASET_ID))
LOOM_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}_{}.loom'.format(TCGA_CODE, DATASET_ID))


########### cleaning the data ############################
df_annotations = pd.read_csv(CELL_ANNOTATIONS_FNAME)
df_annotations['samples'] = df_annotations['samples'].str.upper()
df_annotations.rename(columns={'cell.types': 'cell_type', 'cells': 'cell_id', 'samples': 'sample_id', 'treatment.group': 'treatment_group', 'Cohort': 'cohort'}, inplace=True)
df_annotations['cell_type'] = df_annotations.cell_type.replace({ 'Mal': 'Malignant Cell', 'Endo.': 'Endothelial Cell', 'T.CD4': 'Thelper Cell', 'CAF': 'Fibroblast', 'T.CD8': 'Tcytotoxic Cell', 'T.cell': 'T Cell', 'NK': 'NK Cell', 'B.cell': 'B Cell'})
df_samples = pd.read_excel(SAMPLE_METADATA_FNAME, header=2)
df_samples = df_samples.drop(['Cohort'], axis=1)
df_samples['Sample'] = df_samples.Sample.str.upper()
df_metadata = pd.merge(df_annotations, df_samples, left_on='sample_id', right_on='Sample')
df_metadata = df_metadata.drop(['Sample', 'treatment_group'], axis=1)
df_metadata.rename(columns={'Patient': 'patient_id'})

################### quality control of TPM data ###########################
df_tpm = pd.read_csv(EXP_MTX_TPM_FNAME, index_col=0)
df_tpm.shape
adata = sc.AnnData(X=df_tpm.T.sort_index())
df_obs = df_metadata[['cell_id', 'sample_id', 'cell_type', 'cohort', 'patient_id', 'age', 'sex', 'treatment', 'treatment_group', 'lesion_type', 'site']].set_index('cell_id').sort_index()
adata.obs = df_obs
adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.raw = adata
sc.pp.log1p(adata)
adata.write_h5ad(ANNDATA_FNAME)
adata.to_df().to_csv(EXP_MTX_QC_FNAME) # Write csv file on the disk.

################################ GRN inference default: grnboost2 #########################################
# pyscenic grn pySCENIC_data/results/GSE115978.qc.tpm.csv pySCENIC_data/auxilliaries/lambert2018.txt 
#                   \ -o pySCENIC_data/results/GSE115978.adjacencies.tsv
#                   \ --num_workers 16"