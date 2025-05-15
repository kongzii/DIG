import os, torch, json, ast
import os.path as osp
import ssl
from itertools import repeat
import numpy as np
from rdkit import Chem
import pandas as pd
import networkx as nx
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url

bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}

class SmilesDataset(InMemoryDataset):
    def __init__(self, smiles_path, num_nodes, atom_list, transform=None, pre_transform=None, use_aug=False):
        self.smiles_path = smiles_path
        self.data_name = smiles_path.split("/")[-2]
        self.num_nodes = num_nodes
        self.atom_list = atom_list
        self.use_aug = use_aug
        super(SmilesDataset, self).__init__("/app/data/tmp", transform, pre_transform, force_reload=True)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.smiles_path]

    @property
    def processed_file_names(self):
        return [f'data-{self.data_name}.pt']

    def download(self):
        # No download needed as data is provided directly or via a file path
        pass

    def process(self):
        with open(self.smiles_path, 'r') as f:
            smiles = f.read().splitlines()
        self.smiles = [Chem.CanonSmiles(smile) for smile in smiles]

        data_list = self.pre_process(self.smiles)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = self.data.__class__()
        data.num_nodes = self.num_nodes

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
            
        data['smile'] = self.smiles[idx]
        
        # bfs-searching order
        mol_size = data.num_atom.numpy()[0]
        pure_adj = np.sum(data.adj[:3].numpy(), axis=0)[:mol_size, :mol_size]
        if self.use_aug:
            local_perm = np.random.permutation(mol_size)
            adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
            G = nx.from_numpy_array(np.asmatrix(adj_perm))
            start_idx = np.random.randint(adj_perm.shape[0])
        else:
            local_perm = np.arange(mol_size)
            G = nx.from_numpy_array(np.asmatrix(pure_adj))
            start_idx = 0

        bfs_perm = np.array(self._bfs_seq(G, start_idx))
        bfs_perm_origin = local_perm[bfs_perm]
        bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.num_nodes)])
        data.x = data.x[bfs_perm_origin]
        for i in range(4):
            data.adj[i] = data.adj[i][bfs_perm_origin][:,bfs_perm_origin]
        
        data['bfs_perm_origin'] = torch.Tensor(bfs_perm_origin).long()

        return data

    def pre_process(self, smiles):
        data_list = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            Chem.Kekulize(mol)
            num_atom = mol.GetNumAtoms()

            # atoms
            atom_array = np.zeros((self.num_nodes, len(self.atom_list)), dtype=np.float32)
            atom_idx = 0
            for atom in mol.GetAtoms():
                atom_feature = atom.GetAtomicNum()
                atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1
                atom_idx += 1

            x = torch.tensor(atom_array)

            # bonds
            adj_array = np.zeros([4, self.num_nodes, self.num_nodes], dtype=np.float32)
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                ch = bond_type_to_int[bond_type]
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                adj_array[ch, i, j] = 1.0
                adj_array[ch, j, i] = 1.0
            adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
            adj_array += np.eye(self.num_nodes)

            data = Data(x=x)
            data.adj = torch.tensor(adj_array)
            data.num_atom = num_atom
            data_list.append(data)

        print(f"""
###
Got {len(data_list)} data samples of size {self.num_nodes}.
###
""")

        return data_list
    
    def _bfs_seq(self, G, start_id):
        dictionary = dict(nx.bfs_successors(G, start_id))
        start = [start_id]
        output = [start_id]
        while len(start) > 0:
            next_vertex = []
            while len(start) > 0:
                current = start.pop(0)
                neighbor = dictionary.get(current)
                if neighbor is not None:
                    next_vertex = next_vertex + neighbor
            output = output + next_vertex
            start = next_vertex
        return output
    