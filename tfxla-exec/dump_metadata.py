"""Dump the metadata for the six XLA optimizations in dynamic_dimension_simplifier.cc."""

from pathlib import Path
import json

source_code_dir = Path('source-code-data/tfxla-debug')
ret = {}
for source_code_file in source_code_dir.iterdir():
    optim_file = source_code_file.name
    print(optim_file)
    optim = optim_file[:-3]
    ret[optim] = {
        'hints': [
            {
                'type': 'trigger',
                'codes': [str(source_code_file)]
            }
        ]
    }
with open('optim/tfxla_debug.json', 'w') as f:
    json.dump(ret, f, indent=4)
    