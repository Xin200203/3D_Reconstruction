
import sys
import os
sys.path.append('/datadisk1/xxy/3D_Reconstruction')
try:
    from oneformer3d.instance_merge import OnlineMerge
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
