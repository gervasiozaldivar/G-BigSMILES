# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022-2025: Ludwig Schneider
# See LICENSE for details


def mol_graph_to_rdkit_mol(mol_graph):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is  an optional dependency, but to generate RDKit molecules it is required. Please install RDKit for example with `pip install rdkit`.") from exc

    def convert_bond_type(bond_attr):
        if bond_attr["aromatic"]:
            return Chem.BondType.AROMATIC
        if bond_attr["bond_type"] == 1:
            return Chem.BondType.SINGLE
        if bond_attr["bond_type"] == 2:
            return Chem.BondType.DOUBLE
        if bond_attr["bond_type"] == 3:
            return Chem.BondType.TRIPLE
        if bond_attr["bond_type"] == 4:
            return Chem.BondType.QUADRUPLE

    _DIR_MAP = {"/": Chem.BondDir.ENDUPRIGHT, "\\": Chem.BondDir.ENDDOWNRIGHT}

    mol = Chem.RWMol()
    graph_idx_to_mol_idx = {}
    for graph_idx, data in mol_graph.nodes(data=True):
        atom = Chem.Atom(data["atomic_num"])
        atom.SetIsAromatic(data["aromatic"])
        atom.SetFormalCharge(data["charge"])

        graph_idx_to_mol_idx[graph_idx] = mol.AddAtom(atom)

    for u, v, attr in mol_graph.edges(data=True):
        begin_idx = graph_idx_to_mol_idx[u]
        end_idx = graph_idx_to_mol_idx[v]
        mol.AddBond(begin_idx, end_idx, convert_bond_type(attr))
        bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is None:
            continue
        bond_dir = attr.get("bond_dir", "")
        if bond_dir in _DIR_MAP:
            bond.SetBondDir(_DIR_MAP[bond_dir])

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol
