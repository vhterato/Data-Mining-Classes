import sys
print("Python executable:", sys.executable)

try:
    import numpy as np
    print("numpy:", np.__version__)
except ImportError as e:
    print("numpy error:", e)

try:
    import sklearn
    print("sklearn:", sklearn.__version__)
except ImportError as e:
    print("sklearn error:", e)

try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("matplotlib error:", e)

try:
    import pandas
    print("pandas:", pandas.__version__)
except ImportError as e:
    print("pandas error:", e)
