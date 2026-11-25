import os
import torch
import numpy as np
from sklearn.cluster import KMeans
import argparse


def sample_from_clusters(cluster_ids, S, rng):
    """
    cluster_ids: numpy array, shape (N,)
    S: samples per cluster
    rng: np.random.Generator
    """
    K = cluster_ids.max() + 1
    selected_indices = []

    for c in range(K):
        members = np.where(cluster_ids == c)[0]
        if len(members) <= S:
            chosen = members
        else:
            chosen = rng.choice(members, S, replace=False)
        selected_indices.append(chosen)

    return np.concatenate(selected_indices)


def process_one_file(path, output_dir, K, S, seed):
    print(f"\nProcessing: {path}")

    # 固定随机种子
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1. 加载特征
    X = torch.load(path)
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    else:
        raise ValueError("Loaded .pt file is not a Tensor")

    print(f"Feature shape: {X.shape}")

    # 2. KMeans（固定 random_state）
    print("Running KMeans...")
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(X)

    # 3. 采样（用 rng 控制）
    print("Sampling patches...")
    selected_indices = sample_from_clusters(cluster_ids, S, rng)
    sampled_features = X[selected_indices]

    # 4. 保存
    basename = os.path.basename(path).replace(".pt", "")

    torch.save(torch.tensor(sampled_features), os.path.join(output_dir, basename + ".pt"))
    # torch.save(torch.tensor(selected_indices), os.path.join(save_dir, "selected_indices.pt"))
    # torch.save(torch.tensor(cluster_ids), os.path.join(save_dir, "cluster_ids.pt"))

    print(f"Saved to: {output_dir}")


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    K = args.K
    S = args.S
    seed = args.seed

    os.makedirs(output_dir, exist_ok=True)

    pt_files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    print(f"Found {len(pt_files)} .pt files.")

    for fname in pt_files:
        full_path = os.path.join(input_dir, fname)
        process_one_file(full_path, output_dir, K, S, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="./feature_output")
    parser.add_argument("--output_dir", type=str, default="./cluster_output")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--S", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    main(args)
