import numpy as np
import argparse

COULOMB_CONSTANT = 138.935456
E_INT = 1.0
KJ2KCAL = 1/4.184


def read_xyz_with_charges(xyz_file):
    with open(xyz_file) as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atom_lines = lines[2:2+natoms]

    coords = np.zeros((natoms, 3))
    charges = np.zeros(natoms)

    for i, line in enumerate(atom_lines):
        p = line.split()
        coords[i] = list(map(float, p[1:4]))
        charges[i] = float(p[-1])

    coords *= 0.1  # Å → nm
#    charges /= 18.2223
    return coords, charges


def read_gromacs_index(ndx_file):
    groups = {}
    current = None
    with open(ndx_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                current = line.strip("[]").strip()
                groups[current] = []
            else:
                groups[current].extend([int(i)-1 for i in line.split()])
    return groups


def coulomb_cross(coords, charges, idx1, idx2):
    c1 = coords[idx1]
    q1 = charges[idx1]
    c2 = coords[idx2]
    q2 = charges[idx2]

    diff = c1[:, None, :] - c2[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    qq = q1[:, None] * q2[None, :]

    return COULOMB_CONSTANT * np.sum(qq / r) / E_INT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    coords, charges = read_xyz_with_charges(args.xyz)
    groups = read_gromacs_index(args.index)

    E = coulomb_cross(coords, charges, groups["Protein"], groups["MOL"])
    E *= KJ2KCAL

    with open(args.out, "w") as f:
        f.write("frame,energy_kcal\n")
        f.write(f"{args.xyz},{E:.6f}\n")

