import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import sys
import time
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)

# Configuration imports. Config should be in run_args.py
from thirdparty.default_config.masif_opts import masif_opts

# Load training data (From many files)
from thirdparty.masif_modules.read_data_from_surface import read_data_from_surface, compute_shape_complementarity


def split_ppi_pair_id(ppi_pair_id: str) -> Tuple[str, str, Optional[str]]:
    result = ppi_pair_id.split('_')

    if len(result) == 2:
        return result[0], result[1], None
    elif len(result) == 3:
        return result[0], result[1], result[2]
    else:
        raise ValueError('Invalid ppi_pair_id: {}'.format(ppi_pair_id))


def masif_precompute(ppi_pair_id_list: Tuple[str, str, Optional[str]], output_path: str) -> None: 
    
    np.random.seed(0)
    masif_app = 'masif_site'
    params = masif_opts['site']
    params['ply_chain_dir'] = masif_opts['ply_chain_dir']
    
    precomputation_path = output_path + '/04a-precomputation_9A/precomputation'
    if not os.path.exists(precomputation_path):
        os.makedirs(precomputation_path)
    
    benchmark_surfaces_path = output_path + '/01-benchmark_surfaces'

    print('Reading data from input ply surface files.')
    
    for ppi_pair_id_tuple in ppi_pair_id_list:
        ppi_pair_id, chain1, chain2 = ppi_pair_id_tuple
        
        pdb_path = output_path + '/00-raw_pdbs/' + ppi_pair_id + '.pdb'
        if not os.path.exists(pdb_path):
            continue
        
        my_precomp_dir = precomputation_path + '/' + ppi_pair_id + '_' + chain1 + '/'
        # print(my_precomp_dir)
    
        if not os.path.exists(my_precomp_dir):
            os.makedirs(my_precomp_dir)
        else:
            continue
        
        # Read directly from the ply file.
        if chain2 is None:
            chain2 = ''
        fields = [ppi_pair_id, chain1]
        ply_file = {}
        ply_file['p1'] = f'{benchmark_surfaces_path}/{fields[0]}_{fields[1]}.ply'
        # print('ply_file[p1] = ', ply_file['p1'] )

        if len (fields) == 2 or fields[2] == '':
            pids = ['p1']
        else:
            ply_file['p2'] = f'{benchmark_surfaces_path}/{fields[0]}_{fields[2]}.ply'
            pids = ['p1', 'p2']
        
        # Compute shape complementarity between the two proteins. 
        rho = {}
        neigh_indices = {}
        mask = {}
        input_feat = {}
        theta = {}
        iface_labels = {}
        verts = {}

        for pid in pids:
            input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], iface_labels[pid], verts[pid] = read_data_from_surface(ply_file[pid], params)

        if len(pids) > 1 and masif_app == 'masif_ppi_search':
            start_time = time.time()
            p1_sc_labels, p2_sc_labels = compute_shape_complementarity(ply_file['p1'], ply_file['p2'], neigh_indices['p1'],neigh_indices['p2'], rho['p1'], rho['p2'], mask['p1'], mask['p2'], params)
            np.save(my_precomp_dir+'p1_sc_labels', p1_sc_labels)
            np.save(my_precomp_dir+'p2_sc_labels', p2_sc_labels)
            end_time = time.time()
            print("Computing shape complementarity took {:.2f}".format(end_time-start_time))

        # Save data only if everything went well. 
        for pid in pids: 
            np.save(my_precomp_dir+pid+'_rho_wrt_center', rho[pid])
            np.save(my_precomp_dir+pid+'_theta_wrt_center', theta[pid])
            np.save(my_precomp_dir+pid+'_input_feat', input_feat[pid])
            np.save(my_precomp_dir+pid+'_mask', mask[pid])
            np.save(my_precomp_dir+pid+'_list_indices', neigh_indices[pid])
            np.save(my_precomp_dir+pid+'_iface_labels', iface_labels[pid])
            # Save x, y, z
            np.save(my_precomp_dir+pid+'_X.npy', verts[pid][:,0])
            np.save(my_precomp_dir+pid+'_Y.npy', verts[pid][:,1])
            np.save(my_precomp_dir+pid+'_Z.npy', verts[pid][:,2])
        # time.sleep(30)

    # python $masif_source/data_preparation/04-masif_precompute.py masif_site $PPI_PAIR_ID


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pdb_list', type=str, help='PDB files list')
    argparser.add_argument('--output_path', type=str, help='The output directory to store the data')
    args = argparser.parse_args()

    # Read PDB list
    pdb_list = Path(args.pdb_list).read_text().splitlines()
    ppi_pair_id_list = [split_ppi_pair_id(pdb_name) for pdb_name in pdb_list]
    print(len(ppi_pair_id_list))
    
    # masif precompute
    print(f'MaSIF precomputing at {args.output_path}')
    masif_precompute(ppi_pair_id_list, args.output_path)
    
# python -W ignore 3_masif_precompute.py --pdb_list ./lists/masif_site_only.txt --output_path ./masif_data
