from pathlib import Path
from utils import write_with_subscores
import os

# get target path from command line arg
target_path = os.sys.argv[1]

# Extract the parent directory of target_path (goes two levels up from base)
parent_directory = str(Path(target_path).parent)
dest_path = Path(target_path + "_processed")

# Save counts to the tables directory at the top level, not under base
tables_path = Path(parent_directory) / "tables" / "counts"

# Create directories if they don't exist
dest_path.mkdir(parents=True, exist_ok=True)
tables_path.mkdir(parents=True, exist_ok=True)

jsons_to_process = list(Path(target_path).rglob("**/*.jsonl"))

for item in jsons_to_process:
    filename = item.stem + item.suffix
    filename_ct = item.stem + "_ct.txt"
    filename_ct_path = tables_path / filename_ct
    tgt_file_path = dest_path / filename
    if tgt_file_path.is_file():
        Path.unlink(tgt_file_path)
    write_with_subscores(item, tgt_file_path, filename_ct_path)