import pandas as pd
import numpy as np
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
try:
    cow = pd.options.mode.copy_on_write
    print("copy_on_write:", cow)
except Exception as e:
    print("copy_on_write: not available -", e)

# Check if pandas uses numpy 2.x copy semantics
import sys
print("python:", sys.version)

# Check qlib version
try:
    import qlib
    print("qlib:", getattr(qlib, '__version__', 'unknown'))
except:
    print("qlib: import failed")
