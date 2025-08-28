import os
import pickle
import numpy as np
import shutil
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN  # You can also try KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if __name__=="__main__":
    # Paths
    data_dir = 'E:/data/smplx_multisimo/vit_vec_data/output'         # folder with .jpg and .pkl
    output_dir = 'E:/data/smplx_multisimo/vit_vec_data/cluster'        # output cluster folders
    os.makedirs(output_dir, exist_ok=True)

    # Collect features and filenames
    features = []
    image_paths = []

    for fname in os.listdir(data_dir):
        if fname.endswith('.pkl'):
            base = fname[:-4]
            jpg_path = os.path.join(data_dir, base + '.jpg')
            pkl_path = os.path.join(data_dir, fname)

            if not os.path.exists(jpg_path):
                continue  # skip if image doesn't exist

            with open(pkl_path, 'rb') as f:
                vec = pickle.load(f)  # assumes (1, 768)
                vec = np.squeeze(vec)  # convert to (768,)

            features.append(vec)
            image_paths.append(jpg_path)

    features = np.array(features)
    print(f"Loaded {len(features)} features.")

    # 1. PCA to 128 dims
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=128)
    features_pca = pca.fit_transform(features_scaled)
    print("PCA complete.")

    # 2. t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)
    print("t-SNE complete.")

    # 3. Clustering (DBSCAN, can also try KMeans)
    clustering = DBSCAN(eps=4.0, min_samples=5).fit(features_tsne)
    labels = clustering.labels_  # -1 = outlier

    print(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")

    # 4. Copy images to folders
    for label, img_path in zip(labels, image_paths):
        label_str = f"cluster_{label}" if label != -1 else "outliers"
        cluster_dir = os.path.join(output_dir, label_str)
        os.makedirs(cluster_dir, exist_ok=True)

        dest_path = os.path.join(cluster_dir, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)

    print("Images copied to cluster folders.")

    # tsne visulization
    # Assign a color to each cluster (including outliers)
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # or 'tab20', 'nipy_spectral', etc.

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        idxs = np.where(labels == label)
        color = 'k' if label == -1 else colors(i)
        label_str = 'Outliers' if label == -1 else f'Cluster {label}'
        plt.scatter(
            features_tsne[idxs, 0],
            features_tsne[idxs, 1],
            s=20,
            color=color,
            label=label_str,
            alpha=0.7,
            edgecolors='k' if label != -1 else 'none'
        )

    plt.title('t-SNE of Hand Shape Vectors')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tsne_hand_clusters.png', dpi=300)
    plt.show()
