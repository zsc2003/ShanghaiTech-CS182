import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

import Bio
from Bio.PDB import * 

from thirdparty.default_config.masif_opts import masif_opts
# Local includes
from thirdparty.input_output.protonate import protonate

def split_ppi_pair_id(ppi_pair_id: str) -> Tuple[str, str, Optional[str]]:
    result = ppi_pair_id.split('_')

    if len(result) == 2:
        return result[0], result[1], None
    elif len(result) == 3:
        return result[0], result[1], result[2]
    else:
        raise ValueError(f'Invalid ppi_pair_id: {ppi_pair_id}')


def download_pdb(ppi_pair_id_list: Tuple[str, str, Optional[str]], output_path: str) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # print(masif_opts['raw_pdb_dir'])
    
    pdbl = PDBList()
    for ppi_pair_id in ppi_pair_id_list:
        pdb_filename = pdbl.retrieve_pdb_file(
            ppi_pair_id[0], pdir=output_path, file_format='pdb'
        )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pdb_list', type=str, help='PDB files list')
    argparser.add_argument('--output_path', type=str, help='The output path to store the downloaded PDBs')
    args = argparser.parse_args()

    # Read PDB list
    pdb_list = Path(args.pdb_list).read_text().splitlines()
    ppi_pair_id_list = [split_ppi_pair_id(pdb_name) for pdb_name in pdb_list]
    print(len(ppi_pair_id_list))

    # Download PDB
    print(f'Downloading PDB at {args.output_path}')
    download_pdb(ppi_pair_id_list, args.output_path)

# python 1_download_pdb.py --pdb_list ./lists/masif_site_only.txt --output_path ./masif_data/00-raw_pdbs
