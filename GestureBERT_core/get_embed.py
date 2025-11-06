import os
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from BERTdataset import BERTDataset,custom_collate
from BERTmodel import BERTpretrain, BERT
from BERTtrainer import BERTTrainer

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN,KMeans  # also try KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

if __name__ == "__main__":
    pretrain_model_path = "models/bert_trained.model.ep2500.pth"
    #pretrain_model_path = "models/gesGroup.model.ep500.pth"
    data_dir = "data/multisimo/smplx_poses"
    cache_dir="data/multisimo/caches"
    output_dir="output/embed_cluster"
    os.makedirs(output_dir,exist_ok=True)
    train_kw="after pre-training"
    cluster_kw="KMEANS"
    n_key_frames=4

    n_clusters=10

    seq_len=32 # should be multiples of n_frame_per_file
    n_frame_per_file=16
    batch_size=1
    num_workers=1

    pose_embed_size=454
    pose_mask_size = 62
    hidden=256
    n_layers = 8
    attn_heads=8
    bert_alpha=1.0
    bert_beta=0.2
    bert_gamma=2.0

    epochs=2000
    log_freq=10
    save_freq=100
    lr=1e-4
    warmup_steps=10000
    init_lr_scale=0.5
    adam_weight_decay=0.01
    adam_beta1=0.9
    adam_beta2=0.99

    print(f"Loading Dataset: {data_dir}")
    dataset = BERTDataset(data_dir, seq_len=seq_len,n_frame_file=n_frame_per_file,full_seq=False)

    print("Creating Dataloader")
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,shuffle=False,collate_fn=custom_collate)

    print("Building BERT model")
    bert = BERT(pose_embed_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads)
    # we only need to load bert model weight here to extract sequence representation
    if pretrain_model_path is not None:
        state_dict = torch.load(pretrain_model_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            print(key)
            if key.startswith("bert"):
                new_key=key[5:]
                print(new_key)
                print()
                new_state_dict[new_key] = value
        bert.load_state_dict(new_state_dict)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, pose_embed_size, pose_mask_size,train_dataloader=None, test_dataloader=None,
                          bert_alpha=bert_alpha, bert_beta=bert_beta,bert_gamma=bert_gamma,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          warmup_steps=warmup_steps,init_lr_scale=init_lr_scale,
                          with_cuda=True, log_freq=log_freq)

    print("Getting GestureBERT sequence embeddings...")
    seq_embeds,img_list=trainer.get_embed(data_loader,av_embed=False) #[n_data,hidden]
    av_seq_embeds,_ = trainer.get_embed(data_loader, av_embed=True) #[n_data,hidden]


    print("TSNE fitting...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    seq_embeds_tsne = tsne.fit_transform(seq_embeds)#[n_data,2]
    av_seq_embeds_tsne = tsne.fit_transform(av_seq_embeds)#[n_data,2]

    # Clustering (DBSCAN, can also try KMeans)
    print("Clustering sequence embeddings...")
    #clustering = DBSCAN(eps=10.0, min_samples=3).fit(seq_embeds)
    clustering = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42).fit(seq_embeds)

    cls_labels = clustering.labels_  # -1 = outlier, [n_data,]
    #pos_cls_labels=[label+1 for label in cls_labels]
    counts=np.bincount(cls_labels)
    print(f"CLS cluster: Found {len(set(cls_labels)) - (1 if -1 in cls_labels else 0)} clusters.")
    #av_clustering = DBSCAN(eps=10.0, min_samples=3).fit(av_seq_embeds)
    av_clustering = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42).fit(av_seq_embeds)

    av_labels = av_clustering.labels_  # -1 = outlier, [n_data,]
    #pos_av_labels = [label + 1 for label in av_labels]
    av_counts=np.bincount(av_labels)
    print(f"AV cluster: Found {len(set(av_labels)) - (1 if -1 in av_labels else 0)} clusters.")

    # extract a clip_keyword_list, start_end_list of n_data length
    clip_keyword_list = []
    start_end_list = []
    for img_paths in img_list:
        fns=img_paths
        clip_names = [os.path.basename(fn).split('.')[0] for fn in fns]
        frame_infos = [os.path.basename(fn).split('.')[1] for fn in fns]
        assert len(set(clip_names)) == 1, "Full sequence dataset shouldn't yield sequence from same video!"
        start_frame = int(frame_infos[0].split('_')[0]) - 1
        end_frame = int(frame_infos[-1].split('_')[1])
        clip_keyword_list.append(clip_names[0])
        start_end_list.append((start_frame, end_frame))
    print(len(cls_labels))
    print(len(img_list))

    for features_tsne, labels,counts,keyword in zip([seq_embeds_tsne,av_seq_embeds_tsne],[cls_labels,av_labels],[counts,av_counts],['CLS','AV']):
        # tsne visulization
        # Assign a color to each cluster (including outliers)
        unique_labels = set(labels)
        cmap = plt.get_cmap('nipy_spectral')  # or 'tab20', 'nipy_spectral', etc.
        colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]

        plt.close("all")
        plt.figure(figsize=(10, 8))
        # Ensure limits are set
        x_vals, y_vals = features_tsne[:, 0], features_tsne[:, 1]
        plt.xlim(x_vals.min() - 1, x_vals.max() + 1)
        plt.ylim(y_vals.min() - 1, y_vals.max() + 1)
        cluster_bin={}
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            cluster_bin[label] = count
            idxs = np.where(labels == label)
            color = 'black' if label == -1 else colors[i]
            label_str = f'Outliers: {count}' if label == -1 else f'Cluster {label+1}: {count}'

            # Plot each sample with text instead of scatter dot
            for idx in idxs[0]:
                x, y = features_tsne[idx]
                txt=plt.text(x, y, str(label+1), fontsize=12, color=color, ha='center', va='center', alpha=0.9)
                # change linewidth
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1.5, foreground='white'),
                    path_effects.Normal()
                ])
            # Legend using dummy points
            plt.scatter([], [], color='black' if label == -1 else colors[i], label=label_str)


        # Optional: Legend using dummy points
        #for i, (label, count) in enumerate(zip(unique_labels, counts)):
        #    label_str = f'Outliers: {count}' if label == -1 else f'Cluster {label+1}: {count}'
        #    plt.scatter([], [], color='black' if label == -1 else colors[i], label=label_str)
        """
        for i, (label,count) in enumerate(zip(unique_labels,counts)):
            cluster_bin[label] = count
            idxs = np.where(labels == label)
            color = 'k' if label == -1 else colors[i]
            label_str = f'Outliers: {count}' if label == -1 else f'Cluster {label}: {count}'

            #for idx in idxs[0]:
            #    x,y=features_tsne[idx]
            #    plt.text(x, y, str(label), fontsize=8, color=color, ha='center', va='center', alpha=0.8)

            plt.scatter(
                features_tsne[idxs, 0],
                features_tsne[idxs, 1],
                s=20,
                color=color,
                label=label_str,
                alpha=0.7,
                edgecolors='k' if label != -1 else 'none'
            )
        """
        plt.title(f'{cluster_kw} clustering of {keyword} GestureBERT representations ({train_kw})')
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.legend()
        plt.tight_layout()
        out_fn = os.path.join(output_dir,f"{keyword}_BERT_cluster({train_kw}).png")
        plt.savefig(out_fn, dpi=300)


        # Zipfian test
        # Sort by frequency
        sorted_bin = sorted(cluster_bin.items(), key=lambda x: x[1], reverse=True)
        frequencies = np.array([count for _, count in sorted_bin])
        ranks = np.arange(1, len(frequencies) + 1)
        frequencies_norm = frequencies / np.sum(frequencies)

        # bin chart
        # Cluster size bar plot
        plt.close("all")
        plt.figure(figsize=(8, 5))
        cluster_ids = [f"Cluster {cluster_id+1}" for i, (cluster_id, _) in enumerate(sorted_bin)]
        cluster_sizes = [count for _, count in sorted_bin]
        plt.bar(cluster_ids, cluster_sizes, color='skyblue', edgecolor='black')
        plt.xticks(rotation=45)
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Samples")
        plt.title(f"Cluster Size Distribution of {keyword} GestureBERT representations ({train_kw})")
        plt.tight_layout()
        out_fn = os.path.join(output_dir, f"{keyword}_ClusterHistogram({train_kw}).png")
        plt.savefig(out_fn, dpi=300)

        # Zipf function
        def zipf(x, s, C):
            return C / (x ** s)

        # Fit model
        params, _ = curve_fit(zipf, ranks, frequencies_norm, bounds=(0, [5.0, 1.0]))
        s_fit, C_fit = params
        fitted = zipf(ranks, s_fit, C_fit)

        # R² score
        r2 = r2_score(frequencies_norm, fitted)

        # Plot
        plt.close("all")
        plt.figure(figsize=(8, 6))
        plt.plot(np.log(ranks), np.log(frequencies_norm), 'o-', label='Observed (Clusters)')
        plt.plot(np.log(ranks), np.log(fitted), 'r--', label=f'Zipf Fit (s={s_fit:.2f}, R²={r2:.3f})')
        plt.xlabel("log(Rank)")
        plt.ylabel("log(Frequency)")
        plt.title(f"Zipf Plot of GestureBERT Representation Clusters ({train_kw})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_fn = os.path.join(output_dir, f"{keyword}_Zipf_test({train_kw}).png")
        plt.savefig(out_fn, dpi=300)

    # 4. Copy images to folders
    print("Start copying keypoint images to cluster folders.")
    for i,(label, clip_keyword,(s_frame,e_frame)) in enumerate(zip(cls_labels, clip_keyword_list,start_end_list)):
        label_str = f"cluster_{label}" if label != -1 else "outliers"
        cluster_dir = os.path.join(output_dir, label_str)
        os.makedirs(cluster_dir, exist_ok=True)

        img_paths=[os.path.join(cache_dir,clip_keyword,'images',f"frame_{frame:05d}.jpg") for frame in range(s_frame,e_frame)]
        dest_dir=os.path.join(cluster_dir, f"ges_{i}")
        os.makedirs(dest_dir,exist_ok=True)
        #key_frame_step=len(img_paths)//(n_key_frames-1)
        for j,img_path in enumerate(img_paths):
            #if key_frame_step!=0:
            #    if j % key_frame_step != 0:
            #        continue
            dest_path = os.path.join(dest_dir,os.path.basename(img_path))
            shutil.copy(img_path, dest_path)

        # save gif
        gif_path=os.path.join(cluster_dir, f"ges_{i}","annimation.gif")
        frames= [Image.open(p) for p in img_paths]
        frames[0].save(
            gif_path,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=100,  # duration per frame in ms
            loop=0  # 0 = infinite loop
        )

    print("ALL DONE:)")
