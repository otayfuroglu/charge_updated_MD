#!/usr/bin/env python3
import os
import argparse
import numpy as np
from ase.io import read
from ase.atoms import Atoms
from aimnet.calculators import AIMNet2ASE


def get_charges(atoms: Atoms) -> np.ndarray:
    try:
        q = atoms.get_charges()
        if q is not None and len(q) == len(atoms):
            return np.asarray(q, float)
    except Exception:
        pass

    calc = atoms.calc
    if calc is None:
        raise RuntimeError("Calculator attached değil")

    q = calc.get_property("charges", atoms)
    return np.asarray(q, float)


def write_xyz_with_charges(atoms, charges, outfile, comment):
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()

    with open(outfile, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for s, (x, y, z), q in zip(symbols, pos, charges):
            f.write(f"{s:2s} {x: .8f} {y: .8f} {z: .8f} {q: .6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default="aimnet2")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    atoms = read(args.input)
    atoms.calc = AIMNet2ASE(args.model)

    try:
        energy = atoms.get_potential_energy()  # eV
        charges = get_charges(atoms)

        qtot = charges.sum()
        qmin = charges.min()
        qmax = charges.max()
        status = "OK"

    except Exception as e:
        energy = np.nan
        charges = np.zeros(len(atoms))
        qtot = qmin = qmax = np.nan
        status = "WARN"
        print(f"[WARN] Frame {args.frame}: {e}")

    # comment line
    comment = (
        f"model={args.model}; "
        f"frame={args.frame}; "
        f"status={status}; "
        f"E={energy:.8f} eV; "
        f"Q={qtot:.6f}; "
        f"qmin={qmin:.6f}; "
        f"qmax={qmax:.6f}"
    )

    # output file: aynı isim + _charges
    infile_base = os.path.basename(args.input)
    name, ext = os.path.splitext(infile_base)
    out_xyz = os.path.join(args.outdir, f"{name}_charges{ext}")

    write_xyz_with_charges(atoms, charges, out_xyz, comment)
    print(f"[{status}] frame {args.frame} → {out_xyz}")


if __name__ == "__main__":
    main()

