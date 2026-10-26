"""Compare the small-array port against the pinned DataComp threshold + parquet helper."""
import argparse
import ast
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Tuple

import fsspec
import numpy as np
import pandas as pd

from src.curation import load, score_threshold


PIN = "4a8df1992566ef8334773f7152e1855b1f716162"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    args = parser.parse_args()
    sha = subprocess.check_output(["git","-C",str(args.upstream),"rev-parse","HEAD"],text=True).strip()
    if sha != PIN:
        raise RuntimeError(f"DataComp HEAD must be {PIN}")
    # Use git show on the pinned commit so untracked/local source edits cannot affect this test.
    source = subprocess.check_output(["git","-C",str(args.upstream),"show",
                                     f"{PIN}:baselines/apply_filter.py"],text=True,encoding="utf-8")
    tree = ast.parse(source)
    names = {"get_threshold","load_uids_with_clip_score_helper"}
    tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in tree.body} != names:
        raise RuntimeError("The expected upstream functions could not be located")
    _,_,_,rows,_ = load(5701)
    metadata = pd.DataFrame({"uid":[f"{i:032x}" for i in range(len(rows))],
                             "alignment":[row["align"] for row in rows]})
    # DataComp normally loads many source shards in parallel. Supply this one source shard
    # to get_threshold. The parquet selection, comparison and UInt64 packing are upstream code.
    namespace = {"np":np,"pd":pd,"Any":Any,"Tuple":Tuple,
                 "load_metadata":lambda *args,**kwargs: metadata}
    exec(compile(tree,"DataComp/baselines/apply_filter.py","exec"),namespace)
    threshold = namespace["get_threshold"]("", "alignment", .68, 1)
    if score_threshold(metadata.alignment,.68) != threshold:
        raise AssertionError("Threshold differs from DataComp")
    with tempfile.TemporaryDirectory() as directory:
        parquet = Path(directory)/"scores.parquet"
        metadata.to_parquet(parquet)
        selected = namespace["load_uids_with_clip_score_helper"](
            (fsspec.filesystem("file"),str(parquet)),"alignment",threshold,False)
    expected = set(metadata.index[metadata.alignment >= threshold])
    recovered = {(int(row[0])<<64) | int(row[1]) for row in selected}
    if recovered != expected:
        raise AssertionError("Selected UID sets differ")
    print(f"{sha}: {len(rows)} records; threshold {threshold}; {len(expected)} identical selected IDs")


if __name__=="__main__":
    main()
