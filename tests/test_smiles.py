import numpy as np
import pytest

import gbigsmiles


def test_smiles_parsing(chembl_smi_list):
    for smi in chembl_smi_list:
        if len(smi) > 0:
            smiles_instance = gbigsmiles.BigSmiles.make(smi)
            assert smi == smiles_instance.generate_string(True)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_smiles_weight(n, chembl_smi_list):
    rng = np.random.default_rng()
    no_dot_smi = []
    for smi in chembl_smi_list:
        if "." not in smi and len(smi) > 0:
            no_dot_smi.append(smi)

    for i in range(len(no_dot_smi) // n - 1):
        smis = no_dot_smi[i * n : (i + 1) * n]
        system_string = ""
        total_mw = 0.0
        for smi in smis:
            molw = np.round(rng.uniform(1.0, 1e5), 1)
            system_string += f"{smi}.|{molw}|"
            total_mw += molw
        print(system_string)
        big_smiles = gbigsmiles.BigSmiles.make(system_string)
        for mol in big_smiles.mol_molecular_weight_map:
            print("x", mol, big_smiles.mol_molecular_weight_map[mol])
        assert abs(total_mw - big_smiles.total_molecular_weight) < 1e-6


def _rdkit_mol_from_smiles(smi: str):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    parsed = gbigsmiles.BigSmiles.make(smi)
    atom_graph = parsed.get_generating_graph().get_atom_graph()
    mol_graph = atom_graph.sample_mol_graph()
    return Chem, gbigsmiles.mol_graph_to_rdkit_mol(mol_graph)


def test_alkene_stereo_unspecified():
    Chem, mol = _rdkit_mol_from_smiles("FC=CF")
    double_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(double_bonds) == 1
    assert double_bonds[0].GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)


def test_alkene_stereo_trans():
    Chem, mol = _rdkit_mol_from_smiles("F/C=C/F")
    double_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(double_bonds) == 1
    assert double_bonds[0].GetStereo() == Chem.BondStereo.STEREOE


def test_alkene_stereo_cis():
    Chem, mol = _rdkit_mol_from_smiles("F/C=C\\F")
    double_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(double_bonds) == 1
    assert double_bonds[0].GetStereo() == Chem.BondStereo.STEREOZ
