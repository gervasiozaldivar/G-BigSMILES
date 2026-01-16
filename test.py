import gbigsmiles
import numpy as np
from rdkit import Chem

# BigSMILES with terminal descriptors (empty), repeat-unit descriptors, end-groups (H/H), and fixed length.
# Note: only a single backslash is needed for the directional bond; r-strings would double it.
polymer_text = "[H]{[>][<]C/C=C\\C[>][<]}|uniform(100,100)|[H]"
bs = gbigsmiles.BigSmiles.make(polymer_text)
generating_graph = bs.get_generating_graph()
atom_graph = generating_graph.get_atom_graph()
mol_graph = atom_graph.sample_mol_graph(rng=np.random.default_rng(42))
mol = gbigsmiles.mol_graph_to_rdkit_mol(mol_graph)
print(Chem.MolToSmiles(mol))
