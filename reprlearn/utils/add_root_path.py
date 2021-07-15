# Add ROOT directory to the sys.path.
#
# Usage example:
#   In a jupyter notebook, run this script:
#   %load add_root_path.py
#   Then, we can use the "src" library by:
#   from src.models.GMM import GMM

import os, sys
from pathlib import Path

this_nb_path = Path(os.getcwd())
ROOT = this_nb_path.parent
SRC = ROOT/'src'
paths2add = [this_nb_path, ROOT]

print("Project root: ", str(ROOT))
print('Src folder: ', str(SRC))
print("This nb path: ", str(this_nb_path))


for p in paths2add:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
        print(str(p), "added to the path\n")
        
print(sys.path)