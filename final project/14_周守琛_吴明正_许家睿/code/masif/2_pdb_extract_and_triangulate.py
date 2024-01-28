import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace

# Local includes
from thirdparty.default_config.masif_opts import masif_opts
from thirdparty.triangulation.computeMSMS import computeMSMS
from thirdparty.triangulation.fixmesh import fix_mesh
import pymesh
from thirdparty.input_output.extractPDB import extractPDB
from thirdparty.input_output.save_ply import save_ply
from thirdparty.input_output.read_ply import read_ply
from thirdparty.input_output.protonate import protonate
from thirdparty.triangulation.computeHydrophobicity import computeHydrophobicity
from thirdparty.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from thirdparty.triangulation.computeAPBS import computeAPBS
from thirdparty.triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree


def split_ppi_pair_id(ppi_pair_id: str) -> Tuple[str, str, Optional[str]]:
    result = ppi_pair_id.split('_')

    if len(result) == 2:
        return result[0], result[1], None
    elif len(result) == 3:
        return result[0], result[1], result[2]
    else:
        raise ValueError(f'Invalid ppi_pair_id: {ppi_pair_id}')


def total_extract_and_triangulate(ppi_pair_id_list: Tuple[str, str, Optional[str]], output_path: str) -> None:
    intermediate_path = Path(output_path) / 'intermediate'
    raw_pdb_path = Path(output_path) / '00-raw_pdbs'
    benchmark_pdbs_path = Path(output_path) / '01-benchmark_pdbs'
    benchmark_surfaces_path = Path(output_path) / '01-benchmark_surfaces'
    
    if not intermediate_path.exists():
        intermediate_path.mkdir(parents=True)
    if not benchmark_pdbs_path.exists():
        benchmark_pdbs_path.mkdir(parents=True)
    if not benchmark_surfaces_path.exists():
        benchmark_surfaces_path.mkdir(parents=True)
    
    for ppi_pair_id in ppi_pair_id_list:
        pdb_id, chain1, chain2 = ppi_pair_id

        pdb_filename = raw_pdb_path / f'pdb{pdb_id.lower()}.ent'

        if chain2 is None:
            print('chain2 is None')
            single_extract_and_triangulate(pdb_filename, pdb_id, chain1, output_path)
        else:
            single_extract_and_triangulate(pdb_filename, pdb_id, chain1, output_path)
            single_extract_and_triangulate(pdb_filename, pdb_id, chain2, output_path)


def single_extract_and_triangulate(pdb_filename: str, pdb_id: str, chain_ids1: str, output_path: str) -> None:
    if not os.path.exists(pdb_filename):
        return

    out_pdb_stem = output_path + "intermediate" + "/" + pdb_id + "_" + chain_ids1
    out_ply_stem = output_path + "01-benchmark_surfaces" + "/" + pdb_id + "_" + chain_ids1
    
    if os.path.exists(out_ply_stem + ".ply"):
        return
    
    raw_pdb_path = output_path + '00-raw_pdbs'
    print("pdb_filename = ", pdb_filename)

    protonated_file = raw_pdb_path + "/" + pdb_id + ".pdb"
    # print("protonated_file = ", protonated_file)

    protonate(pdb_filename, protonated_file)

    # delete the file 'pdb_filename'
    # os.remove(pdb_filename)

    pdb_filename = protonated_file

    # Extract chains of interest.
    
    
    extractPDB(pdb_filename, out_pdb_stem + ".pdb", chain_ids1)

    # Compute MSMS of surface w/hydrogens, 
    try:
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_pdb_stem + ".pdb", protonate=True)
    except:
        set_trace()

    # Compute "charged" vertices
    if masif_opts['use_hbond']:
        vertex_hbond = computeCharges(out_pdb_stem, vertices1, names1)

    # For each surface residue, assign the hydrophobicity of its amino acid. 
    if masif_opts['use_hphob']:
        vertex_hphobicity = computeHydrophobicity(names1)

    # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
    vertices2 = vertices1
    faces2 = faces1

    # Fix the mesh.
    mesh = pymesh.form_mesh(vertices2, faces2)
    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    # Assign charges on new vertices based on charges of old vertices (nearest
    # neighbor)

    if masif_opts['use_hbond']:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hbond, masif_opts)

    if masif_opts['use_hphob']:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hphobicity, masif_opts)

    if masif_opts['use_apbs']:
        vertex_charges = computeAPBS(regular_mesh.vertices, out_pdb_stem + ".pdb", out_pdb_stem)

    iface = np.zeros(len(regular_mesh.vertices))
    if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
        # Compute the surface of the entire complex and from that compute the interface.
        v3, f3, _, _, _ = computeMSMS(pdb_filename,protonate=True)
        # Regularize the mesh
        mesh = pymesh.form_mesh(v3, f3)
        full_regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        # Find the vertices that are in the iface.
        v3 = full_regular_mesh.vertices
        # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
        kdt = KDTree(v3)
        d, r = kdt.query(regular_mesh.vertices)
        d = np.square(d) # Square d, because this is how it was in the pyflann version.
        assert(len(d) == len(regular_mesh.vertices))
        iface_v = np.where(d >= 2.0)[0]
        iface[iface_v] = 1.0
        # Convert to ply and save.
        save_ply(out_ply_stem + ".ply", regular_mesh.vertices,\
                            regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                            normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                            iface=iface)

    else:
        # Convert to ply and save.
        save_ply(out_ply_stem + ".ply", regular_mesh.vertices,\
                            regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                            normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
    # if not os.path.exists(masif_opts['ply_chain_dir']):
    #     os.makedirs(masif_opts['ply_chain_dir'])
    # if not os.path.exists(masif_opts['pdb_chain_dir']):
    #     os.makedirs(masif_opts['pdb_chain_dir'])
    # shutil.copy(out_ply_stem + '.ply', masif_opts['ply_chain_dir']) 
    # shutil.copy(out_pdb_stem + '.pdb', masif_opts['pdb_chain_dir']) 

    benchmark_path = output_path + "01-benchmark_pdbs"
    intermediate_path = output_path + "intermediate"
    for file in os.listdir(intermediate_path):
        if file.endswith(".pdb"):
            shutil.move(intermediate_path + "/" + file, benchmark_path + "/" + file)
        else:
            os.remove(intermediate_path + "/" + file)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pdb_list', type=str, help='PDB files list')
    argparser.add_argument('--output_path', type=str, help='The output directory to store the data')
    args = argparser.parse_args()

    # Read PDB list
    pdb_list = Path(args.pdb_list).read_text().splitlines()
    ppi_pair_id_list = [split_ppi_pair_id(pdb_name) for pdb_name in pdb_list]
    print(len(ppi_pair_id_list))

    # extract and triangulate pdb
    print(f'Extracting and triangulating PDB at {args.output_path}')
    total_extract_and_triangulate(ppi_pair_id_list, args.output_path)
    
# python -W ignore 2_pdb_extract_and_triangulate.py --pdb_list ./lists/masif_site_only.txt --output_path ./masif_data/
