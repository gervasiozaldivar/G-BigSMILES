import numpy as np
import pytest

import gbigsmiles
import warnings


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


def _rdkit_mol_from_bigsmiles(text: str,seed:int):
    pytest.importorskip("rdkit")
    warnings.filterwarnings("ignore") # there's a warnings due to the alkene in the backbone that's not relevant
    from rdkit import Chem

    parsed = gbigsmiles.BigSmiles.make(text)
    atom_graph = parsed.get_generating_graph().get_atom_graph()
    mol_graph = atom_graph.sample_mol_graph(rng=np.random.default_rng(seed))
    return Chem, gbigsmiles.mol_graph_to_rdkit_mol(mol_graph)


def test_polymer_alkene_stereo_unspecified():
    warnings.filterwarnings("ignore") # there's a warnings due to the alkene in the backbone that's not relevant
    text = "[H]{[>][<]C=CC[>][<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_bigsmiles(text, seed=0)
    doubles = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(doubles) >= 1
    assert all(db.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY) for db in doubles)

def test_polymer_alkene_stereo_trans():
    warnings.filterwarnings("ignore") # there's a warnings due to the alkene in the backbone that's not relevant
    text = "[H]{[>][<]C/C=C/C[>][<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_bigsmiles(text, seed=1)
    doubles = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(doubles) >= 1
    assert any(db.GetStereo() == Chem.BondStereo.STEREOE for db in doubles)


def test_polymer_alkene_stereo_trans_reverse():
    warnings.filterwarnings("ignore") # there's a warnings due to the alkene in the backbone that's not relevant
    text = "[H]{[>][<]C\\C=C\\C[>][<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_bigsmiles(text, seed=2)
    doubles = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(doubles) >= 1
    assert any(db.GetStereo() == Chem.BondStereo.STEREOE for db in doubles)


def test_polymer_alkene_stereo_cis():
    warnings.filterwarnings("ignore") # there's a warnings due to the alkene in the backbone that's not relevant
    text = "[H]{[>][<]C\\C=C/C[>][<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_bigsmiles(text, seed=2)
    doubles = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(doubles) >= 1
    assert any(db.GetStereo() == Chem.BondStereo.STEREOZ for db in doubles)


def test_polymer_alkene_stereo_cis_reverse():
    warnings.filterwarnings("ignore") # there's a warnings due to the alkene in the backbone that's not relevant
    text = "[H]{[>][<]C/C=C\\C[>][<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_bigsmiles(text, seed=2)
    doubles = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(doubles) >= 1
    assert any(db.GetStereo() == Chem.BondStereo.STEREOZ for db in doubles)

def test_polymer_chiral_center_isotactic_polypropylene():
    """Test that atom-level chirality ([C@H]) is preserved in isotactic polypropylene (iPP)."""
    text = "[H]{[>][<]C[C@H](C)[>][<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_bigsmiles(text, seed=0)
    # Find chiral carbon atoms (those with 4 different substituents including H)
    chiral_atoms = [a for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    # iPP should have chiral centers at the methine carbons
    assert len(chiral_atoms) >= 1, "Expected at least one chiral center in isotactic polypropylene"
    # All specified stereocenters should be the same configuration (isotactic)
    chiral_tags = [a.GetChiralTag() for a in chiral_atoms]
    assert all(tag == chiral_tags[0] for tag in chiral_tags), "All stereocenters should have same configuration (isotactic)"

