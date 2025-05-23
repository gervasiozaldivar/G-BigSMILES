atom_name_mapping = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    -1: "*",
}

atom_name_num = dict((v, k) for k, v in atom_name_mapping.items())
# Add aromatic versions
atom_name_num["c"] = atom_name_num["C"]
atom_name_num["b"] = atom_name_num["B"]
atom_name_num["n"] = atom_name_num["N"]
atom_name_num["o"] = atom_name_num["O"]
atom_name_num["s"] = atom_name_num["S"]
atom_name_num["p"] = atom_name_num["P"]


atom_color_mapping = {
    -1: "FFFFFF",
    1: "FFFFFF",
    2: "D9FFFF",
    3: "CC80FF",
    4: "C2FF00",
    5: "FFB5B5",
    6: "909090",
    7: "105050",
    8: "FF0D0D",
    9: "90E050",
    10: "B3E3F5",
    11: "AB5CF2",
    12: "8AFF00",
    13: "BFA6A6",
    14: "F0C8A0",
    15: "FF8000",
    16: "FFFF30",
    17: "1FF01F",
    18: "80D1E3",
    19: "8F40D4",
    20: "3DFF00",
    21: "E6E6E6",
    22: "BFC2C7",
    23: "A6A6AB",
    24: "8A99C7",
    25: "9C7AC7",
    26: "E06633",
    27: "F090A0",
    28: "50D050",
    29: "C88033",
    30: "7D80B0",
    31: "C28F8F",
    32: "668F8F",
    33: "BD80E3",
    34: "FFA100",
    35: "A62929",
    36: "5CB8D1",
    37: "702EB0",
    38: "00FF00",
    39: "94FFFF",
    40: "94E0E0",
    41: "73C2C9",
    42: "54B5B5",
    43: "3B9E9E",
    44: "248F8F",
    45: "0A7D8C",
    46: "006985",
    47: "C0C0C0",
    48: "FFD98F",
    49: "A67573",
    50: "668080",
    51: "9E63B5",
    52: "D47A00",
    53: "940094",
    54: "429EB0",
    55: "57178F",
    56: "00C900",
    57: "70D4FF",
    58: "FFFFC7",
    59: "D9FFC7",
    60: "C7FFC7",
    61: "A3FFC7",
    62: "8FFFC7",
    63: "61FFC7",
    64: "45FFC7",
    65: "30FFC7",
    66: "1FFFC7",
    67: "00FF9C",
    68: "00E675",
    69: "00D452",
    70: "00BF38",
    71: "00AB24",
    72: "4DC2FF",
    73: "4DA6FF",
    74: "2194D6",
    75: "267DAB",
    76: "266696",
    77: "175487",
    78: "D0D0E0",
    79: "FFD123",
    80: "B8B8D0",
    81: "A6544D",
    82: "575961",
    83: "9E4FB5",
    84: "AB5C00",
    85: "754F45",
    86: "428296",
    87: "420066",
    88: "007D00",
    89: "70ABFA",
    90: "00BAFF",
    91: "00A1FF",
    92: "008FFF",
    93: "0080FF",
    94: "006BFF",
    95: "545CF2",
    96: "785CE3",
    97: "8A4FE3",
    98: "A136D4",
    99: "B31FD4",
    100: "B31FBA",
    101: "B30DA6",
    102: "BD0D87",
    103: "C70066",
    104: "CC0059",
    105: "D1004F",
    106: "D90045",
    107: "E00038",
    108: "E6002E",
    109: "EB0026",
}

atomic_masses = {
    1: 1.008,
    2: 4.0026,
    3: 6.94,
    4: 9.0122,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    10: 20.180,
    11: 22.990,
    12: 24.305,
    13: 26.982,
    14: 28.085,
    15: 30.974,
    16: 32.06,
    17: 35.45,
    18: 39.948,
    19: 39.098,
    20: 40.078,
    21: 44.956,
    22: 47.867,
    23: 50.942,
    24: 51.996,
    25: 54.938,
    26: 55.845,
    27: 58.933,
    28: 58.693,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.63,
    33: 74.922,
    34: 78.971,
    35: 79.904,
    36: 83.798,
    37: 85.468,
    38: 87.62,
    39: 88.906,
    40: 91.224,
    41: 92.906,
    42: 95.95,
    43: 98,
    44: 101.07,
    45: 102.91,
    46: 106.42,
    47: 107.87,
    48: 112.41,
    49: 114.82,
    50: 118.71,
    51: 121.76,
    52: 127.60,
    53: 126.90,
    54: 131.29,
    55: 132.91,
    56: 137.33,
    57: 138.91,
    58: 140.12,
    59: 140.91,
    60: 144.24,
    61: 145,
    62: 150.36,
    63: 151.96,
    64: 157.25,
    65: 158.93,
    66: 162.50,
    67: 164.93,
    68: 167.26,
    69: 168.93,
    70: 173.05,
    71: 174.97,
    72: 178.49,
    73: 180.95,
    74: 183.84,
    75: 186.21,
    76: 190.23,
    77: 192.22,
    78: 195.08,
    79: 196.97,
    80: 200.59,
    81: 204.38,
    82: 207.2,
    83: 208.98,
    84: 209,
    85: 210,
    86: 222,
    87: 223,
    88: 226,
    89: 227,
    90: 232.04,
    91: 231.04,
    92: 238.03,
    93: 237,
    94: 244,
    95: 243,
    96: 247,
    97: 247,
    98: 251,
    99: 252,
    100: 257,
}

default_valence = {
    1: 1,
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: -3,
    8: -2,
    9: -1,
    10: 0,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 5,
    16: -2,
    17: -1,
    18: 0,
    19: 1,
    20: 2,
    21: 3,
    22: 4,
    23: 5,
    24: 6,
    25: 2,
    26: 3,
    27: 2,
    28: 2,
    29: 2,
    30: 2,
    31: 3,
    32: 4,
    33: 3,
    34: 4,
    35: -1,
    36: 0,
    37: 1,
    38: 2,
    39: 3,
    40: 4,
    41: 5,
    42: 6,
    43: 7,
    44: 4,
    45: 3,
    46: 4,
    47: 1,
    48: 2,
    49: 3,
    50: 4,
    51: 3,
    52: 4,
    53: -1,
    54: 0,
    55: 1,
    56: 2,
    57: 3,
    58: 4,
    59: 3,
    60: 3,
    61: 3,
    62: 3,
    63: 3,
    64: 3,
    65: 3,
    66: 3,
    67: 3,
    68: 3,
    69: 3,
    70: 3,
    71: 3,
    72: 4,
    73: 5,
    74: 6,
    75: 7,
    76: 8,
    77: 4,
    78: 4,
    79: 3,
    80: 2,
    81: 3,
    82: 4,
    83: 3,
    84: 4,
    85: -1,
    86: 0,
    87: 1,
    88: 2,
    89: 3,
    90: 4,
    91: 5,
    92: 6,
    93: 7,
    94: 4,
    95: 3,
    96: 3,
    97: 3,
    98: 3,
    99: 3,
    100: 3,
}

smi_bond_mapping = {
    ".": 0,
    "-": 1,
    "=": 2,
    "#": 3,
    "$": 4,
}

bond_num_smi = dict((v, k) for k, v in smi_bond_mapping.items())
