"""Generate scene .npz files for XoFTR evaluation from a folder of images.

Usage examples:

  # Minimal: visible + thermal folders with calibration YAML
  python create_scene_npz.py \
      --vis_dir  data/my_scene/visible \
      --tir_dir  data/my_scene/thermal \
      --calib    data/my_scene/calibration.yaml \
      --out_dir  data/my_scene/index

  # With ground-truth poses (required for AUC evaluation)
  python create_scene_npz.py \
      --vis_dir  data/my_scene/visible \
      --tir_dir  data/my_scene/thermal \
      --calib    data/my_scene/calibration.yaml \
      --poses    data/my_scene/poses.txt \
      --out_dir  data/my_scene/index

  # Control pairing: stride and max pairs
  python create_scene_npz.py \
      --vis_dir  data/my_scene/visible \
      --tir_dir  data/my_scene/thermal \
      --calib    data/my_scene/calibration.yaml \
      --out_dir  data/my_scene/index \
      --pair_stride 1 \
      --max_pairs 500

Calibration YAML format:
  visible:
    fx: 2948.12
    fy: 2940.02
    cx: 1929.92
    cy: 1072.14
    dist: [0.271, -0.932, -0.00238, -0.00225, 0.891]
  thermal:
    fx: 768.62
    fy: 766.14
    cx: 316.95
    cy: 248.34
    dist: [-0.343, -0.0215, 0.000615, -0.00097, 0.365]

Poses file format (one line per image, space-separated):
  filename 4x4_matrix_row_major (16 values)
  IM_00001.jpg 0.922 -0.069 -0.381 6.018 0.095 0.994 0.048 0.531 ...

If no poses are provided, identity poses are used. Pair evaluation
(AUC metrics) will be meaningless, but matching visualization still works.
"""

import argparse
import os
import numpy as np
import yaml
from pathlib import Path
from itertools import combinations


def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)

    result = {}
    for sensor in ['visible', 'thermal']:
        c = calib[sensor]
        K = np.array([
            [c['fx'], 0.0,    c['cx']],
            [0.0,     c['fy'], c['cy']],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)

        dist_raw = c.get('dist', [0.0] * 5)
        dist = np.zeros(8, dtype=np.float64)
        dist[:len(dist_raw)] = dist_raw

        result[sensor] = {'K': K, 'dist': dist}

    return result


def load_poses(poses_path, image_names):
    """Load 4x4 poses from a text file. Returns dict mapping filename -> 4x4 matrix."""
    poses = {}
    with open(poses_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            name = parts[0]
            mat = np.array([float(x) for x in parts[1:17]]).reshape(4, 4)
            poses[name] = mat

    result = {}
    for name in image_names:
        basename = os.path.basename(name)
        stem = os.path.splitext(basename)[0]
        if basename in poses:
            result[name] = poses[basename]
        elif stem in poses:
            result[name] = poses[stem]
        else:
            result[name] = np.eye(4, dtype=np.float64)
    return result


def find_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = sorted([
        f.name for f in Path(directory).iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ])
    return images


def generate_pairs(n_images, stride=1, max_pairs=None):
    if stride <= 0:
        pairs = list(combinations(range(n_images), 2))
    else:
        pairs = []
        for i in range(n_images):
            for j in range(i + stride, n_images, stride):
                pairs.append((i, j))

    if max_pairs and len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Generate scene .npz files for XoFTR evaluation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--vis_dir', required=True,
                        help='Directory containing visible-spectrum images')
    parser.add_argument('--tir_dir', required=True,
                        help='Directory containing thermal images')
    parser.add_argument('--calib', required=True,
                        help='Path to calibration YAML file')
    parser.add_argument('--poses', default=None,
                        help='Path to poses text file (optional)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for npz and list files')
    parser.add_argument('--scene_name', default=None,
                        help='Scene name for the npz file (default: parent dir name)')
    parser.add_argument('--data_root', default=None,
                        help='Data root directory. Image paths in the npz are stored '
                             'relative to this. Default: parent of vis_dir parent.')
    parser.add_argument('--pair_stride', type=int, default=1,
                        help='Pair every image with every stride-th neighbor. '
                             '0 = all combinations. Default: 1')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum number of pairs (randomly sampled if exceeded)')

    args = parser.parse_args()

    vis_dir = Path(args.vis_dir).resolve()
    tir_dir = Path(args.tir_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    vis_images = find_images(vis_dir)
    tir_images = find_images(tir_dir)

    if not vis_images:
        raise FileNotFoundError(f"No images found in {vis_dir}")
    if not tir_images:
        raise FileNotFoundError(f"No images found in {tir_dir}")

    matched = sorted(set(vis_images) & set(tir_images))
    vis_only = sorted(set(vis_images) - set(tir_images))
    tir_only = sorted(set(tir_images) - set(vis_images))

    if matched:
        print(f"Found {len(matched)} matched vis/tir image pairs (same filenames)")
        image_names = matched
    else:
        if len(vis_images) != len(tir_images):
            print(f"WARNING: Different number of images: {len(vis_images)} visible, "
                  f"{len(tir_images)} thermal. Using min count and pairing by sort order.")
        n = min(len(vis_images), len(tir_images))
        image_names = vis_images[:n]
        tir_images_used = tir_images[:n]
        print(f"No matching filenames found. Pairing {n} images by sorted order.")

    if vis_only:
        print(f"  {len(vis_only)} visible-only images skipped")
    if tir_only:
        print(f"  {len(tir_only)} thermal-only images skipped")

    n_images = len(image_names)

    if args.data_root:
        data_root = Path(args.data_root).resolve()
    else:
        data_root = vis_dir.parent.parent
        if not vis_dir.is_relative_to(data_root):
            data_root = vis_dir.parent

    vis_rel = vis_dir.relative_to(data_root)
    tir_rel = tir_dir.relative_to(data_root)

    calib = load_calibration(args.calib)

    image_paths = np.empty((n_images, 2), dtype=f'<U{256}')
    intrinsics = np.empty((n_images, 2, 3, 3), dtype=np.float64)
    distortion_coefs = np.empty((n_images, 2, 8), dtype=np.float64)
    poses = np.empty((n_images, 4, 4), dtype=np.float64)

    if matched:
        tir_names = matched
    else:
        tir_names = tir_images_used

    for i, vis_name in enumerate(image_names):
        tir_name = tir_names[i]
        image_paths[i, 0] = str(vis_rel / vis_name).replace('\\', '/')
        image_paths[i, 1] = str(tir_rel / tir_name).replace('\\', '/')
        intrinsics[i, 0] = calib['visible']['K']
        intrinsics[i, 1] = calib['thermal']['K']
        distortion_coefs[i, 0] = calib['visible']['dist']
        distortion_coefs[i, 1] = calib['thermal']['dist']

    if args.poses:
        pose_dict = load_poses(args.poses, image_names)
        for i, name in enumerate(image_names):
            poses[i] = pose_dict[name]
        print(f"Loaded poses from {args.poses}")
    else:
        for i in range(n_images):
            poses[i] = np.eye(4)
        print("No poses provided — using identity (AUC metrics will be meaningless)")

    pairs = generate_pairs(n_images, args.pair_stride, args.max_pairs)
    pair_infos = np.array(pairs, dtype=object)
    print(f"Generated {len(pairs)} pairs from {n_images} images "
          f"(stride={args.pair_stride}, max={args.max_pairs})")

    scene_name = args.scene_name or vis_dir.parent.name
    npz_dir = out_dir / 'scene_info_test'
    list_dir = out_dir / 'val_test_list'
    npz_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    npz_filename = f'{scene_name}.npz'
    npz_path = npz_dir / npz_filename
    np.savez(npz_path,
             image_paths=image_paths,
             intrinsics=intrinsics,
             distortion_coefs=distortion_coefs,
             poses=poses,
             pair_infos=pair_infos)
    print(f"Saved {npz_path}")

    list_path = list_dir / 'test_list.txt'
    with open(list_path, 'w') as f:
        f.write(npz_filename + '\n')
    print(f"Saved {list_path}")

    print(f"\nTo run evaluation:")
    print(f"  python test_relative_pose.py xoftr --save_figs "
          f"--data_root_dir {data_root}")


if __name__ == '__main__':
    main()
