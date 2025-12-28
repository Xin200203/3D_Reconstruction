#!/usr/bin/env python3
import sys
import numpy as np

def load_obj_points(path):
    pts = []
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            if len(parts) >= 7:
                r, g, b = map(float, parts[4:7])
            else:
                r, g, b = 200.0, 200.0, 200.0
            pts.append([x, y, z])
            cols.append([r, g, b])
    pts = np.asarray(pts, dtype=np.float32)
    cols = np.asarray(cols, dtype=np.float32)
    if cols.size and cols.max() > 1.0:
        cols /= 255.0
    return pts, cols

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_obj_points.py /path/to/points.obj")
        sys.exit(1)
    path = sys.argv[1]

    import open3d as o3d

    pts, cols = load_obj_points(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 3.0
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
