"""Build the UniRef100 accession-to-taxonomy map used for MSA pairing.

Taxonomy identifiers are parsed from a ColabFold/MMseqs2
``uniref30_*_db_h`` header database. Records in that file are
null-delimited.
"""

import argparse
import pickle
import re
from pathlib import Path

HEADER_RE = re.compile(rb"^UniRef100_(\S+).*?TaxID=(\d+)")


def main(args: argparse.Namespace) -> None:
    """Parse the header DB and dump the accession -> taxid pickle."""
    mapping: dict[str, int] = {}
    n_total = 0
    n_matched = 0

    with args.header_db.open("rb") as f:
        buf = f.read()

    for rec in buf.split(b"\x00"):
        if not rec:
            continue
        n_total += 1
        m = HEADER_RE.match(rec)
        if m:
            n_matched += 1
            mapping[m.group(1).decode("ascii")] = int(m.group(2))

    print(f"Parsed {n_total} header records; {n_matched} had TaxID ({100*n_matched/max(n_total,1):.1f}%).")  # noqa: T201
    print(f"Unique accessions mapped: {len(mapping)}")  # noqa: T201

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build UniRef100 accession -> TaxID map.")
    parser.add_argument("--header-db", type=Path, required=True, help="Path to *_db_h file.")
    parser.add_argument("--out", type=Path, required=True, help="Output pickle path.")
    args = parser.parse_args()
    main(args)
