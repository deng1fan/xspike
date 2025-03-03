
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_distribution_curve(
    a_lists, a_labels, b_lists, b_labels, xlabel, ylabel, file_name
):
    # Initialize the plot size
    plt.figure(figsize=(10, 6))

    # Generate colors from a Seaborn color palette
    palette = sns.color_palette("viridis", len(a_lists))

    # Plot solid lines for a_lists
    for a, color, label in zip(a_lists, palette, a_labels):
        # Sort a scores
        a = sorted(a)
        total_count = len(a)

        # Generate thresholds and calculate proportions
        thresholds = np.linspace(min(a), max(a), 200)
        proportions = [
            (np.array(a) > threshold).sum() / total_count for threshold in thresholds
        ]

        # Apply a rolling mean for smoothing
        proportions_smooth = (
            pd.Series(proportions).rolling(window=5, center=True).mean()
        )

        # Plot the smoothed curve with a solid line
        sns.lineplot(
            x=thresholds, y=proportions_smooth, color=color, linewidth=2, label=label
        )

    # Generate colors from a Seaborn color palette
    palette = sns.color_palette("viridis", len(a_lists))
    # Plot dashed lines for b_lists
    for b, color, label in zip(b_lists, palette, b_labels):
        # Sort b scores
        b = sorted(b)
        total_count = len(b)

        # Generate thresholds and calculate proportions
        thresholds = np.linspace(min(b), max(b), 200)
        proportions = [
            (np.array(b) > threshold).sum() / total_count for threshold in thresholds
        ]

        # Apply a rolling mean for smoothing
        proportions_smooth = (
            pd.Series(proportions).rolling(window=5, center=True).mean()
        )

        # Append " (Dashed)" to the label for b_lists
        sns.lineplot(
            x=thresholds,
            y=proportions_smooth,
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{label} (Dashed)",
        )

    # Set axis labels with bold font
    plt.xlabel(xlabel, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")

    # Adjust legend and invert x-axis
    plt.gca().invert_xaxis()

    # Save the plot as a high-resolution PDF
    plt.savefig(file_name, format="pdf", dpi=300)
    plt.close()



def visualize_normsim_multi(texts_sets, refs, labels, batch_size=32, file_name="normsim_multi_plot.png"):
    
    def embed_texts(texts, batch_size):
        # 初始化 BERT 模型和分词器
        tokenizer = BertTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model = BertModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to("cuda")
        with torch.no_grad():   
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=514,
                ).to("cuda")
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def plot_similarity_distribution_curve(similarity_lists, labels, file_name):
        # 设置颜色和标签列表
        colors = plt.cm.viridis(np.linspace(0, 1, len(similarity_lists)))
        
        plt.figure(figsize=(10, 6))
        
        for similarity_list, color, label in zip(similarity_lists, colors, labels):
            # 排序相似度列表
            similarity_list = sorted(similarity_list)
            total_count = len(similarity_list)
            
            # 生成更密集的阈值范围，从最小值到最大值
            thresholds = np.linspace(min(similarity_list), max(similarity_list), 200)
            proportions = [(np.array(similarity_list) > threshold).sum() / total_count for threshold in thresholds]
            # 将 proportions 转为 pandas Series，应用移动平均
            proportions_smooth = pd.Series(proportions).rolling(window=5, center=True).mean()

            # 绘制平滑的曲线，并添加图注
            plt.plot(thresholds, proportions_smooth, color=color, linewidth=2, label=label)
        
        plt.xlabel("Similarity Threshold", fontsize=14, fontweight='bold')
        plt.ylabel("Proportion of Datapoints", fontsize=14, fontweight='bold')
        # plt.title("Proportion of Similarity Scores Above Threshold for Multiple Text Sets")
        # plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis()
        plt.savefig(file_name)
        plt.close()

    ref_embeddings = embed_texts(refs, batch_size)
    all_normalized_data = []

    for texts in texts_sets:
        texts_embeddings = embed_texts(texts, batch_size)
        sims = []
        for t in texts_embeddings:
            max_sim = -np.inf
            for z in ref_embeddings:
                similarity = np.dot(z, t)
                if similarity > max_sim:
                    max_sim = similarity
            sims.append(max_sim)
        
        sims = np.array(sims)
        min_val = np.min(sims)
        max_val = np.max(sims)
        normalized_data = (sims - min_val) / (max_val - min_val)
        all_normalized_data.append(normalized_data)
    
    plot_similarity_distribution_curve(all_normalized_data, labels, file_name)
    return all_normalized_data
        



def visualize_text_clusters(
    texts,
    model_name="bert-base-uncased",
    max_clusters=10,
    fix_clusters=-1,
    perplexity=30,
    max_iter=1000,
    random_state=42,
    file_name="clusters_plot.png",
    batch_size=32,
):
    """
    将文本数据嵌入到高维空间，使用 t-SNE 进行降维，使用 KMeans 进行聚类，并绘制和保存聚类结果。

    参数:
    texts (list of str): 输入的文本列表。
    model_name (str): 使用的预训练模型名称，默认为 'bert-base-uncased'。
    max_clusters (int): 最大聚类簇数，默认为 10。
    fix_clusters (int): 固定聚类簇数，默认为 -1。 如果为 -1，则根据 silhouette 指标选择最佳的聚类簇数。否则，使用 fix_clusters 指定的簇数。
    perplexity (int): t-SNE 中的困惑度参数，默认为 30。
    max_iter (int): t-SNE 的迭代次数，默认为 1000。
    random_state (int): 随机种子，默认为 42。
    file_name (str): 保存图片的文件名，默认为 'clusters_plot.png'。
    batch_size (int): 处理文本嵌入的批量大小，默认为 32。

    返回:
    list of dict: 每个元素包含中心数据索引和簇内所有数据的索引。
    """

    def embed_texts(texts, model_name, batch_size):
        # 初始化 BERT 模型和分词器
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to("cuda")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def find_nearest_center(cluster_points, original_indices, center):
        distances = cdist(cluster_points, [center])
        nearest_idx = np.argmin(distances)
        return original_indices[nearest_idx]

    def optimal_clusters(data, max_clusters):
        best_n_clusters = 2
        best_silhouette = -1

        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=random_state)
            labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)

            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n

        return best_n_clusters

    def tsne_cluster(
        data,
        original_indices,
        max_clusters=10,
        fix_clusters=-1,
        perplexity=30,
        max_iter=1000,
        random_state=42,
    ):
        data = np.array(data)
        if data.ndim != 2:
            raise ValueError("输入数据必须为二维数组")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
        )
        tsne_result = tsne.fit_transform(data)

        optimal_n_clusters = optimal_clusters(tsne_result, max_clusters) if fix_clusters == -1 else fix_clusters
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(tsne_result)

        unique_labels = set(labels)
        clusters = []
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_points = tsne_result[cluster_indices]
            center = cluster_points.mean(axis=0)
            nearest_center_index = find_nearest_center(
                cluster_points, original_indices[cluster_indices], center
            )
            cluster_info = {
                "center_index": nearest_center_index,
                "cluster_indices": original_indices[cluster_indices].tolist(),
            }
            clusters.append(cluster_info)

        return clusters

    def plot_and_save_clusters(clusters, tsne_result, labels, file_name):
        plt.figure(figsize=(10, 8))
        colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

        for cluster_id, cluster_data in enumerate(clusters):
            data = tsne_result[np.array(cluster_data["cluster_indices"])]
            center = tsne_result[cluster_data["center_index"]]
            plt.scatter(
                data[:, 0],
                data[:, 1],
                c=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id}",
            )
            plt.scatter(center[0], center[1], c="black", marker="x", s=100)

        plt.title("t-SNE Clustering")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True)

        plt.savefig(file_name)
        plt.close()

    embeddings = embed_texts(texts, model_name, batch_size)
    original_indices = np.arange(len(texts))
    clusters = tsne_cluster(
        embeddings,
        original_indices,
        max_clusters=max_clusters,
        fix_clusters=fix_clusters,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    tsne_result = tsne.fit_transform(embeddings)
    labels = KMeans(n_clusters=len(clusters), random_state=random_state).fit_predict(
        tsne_result
    )

    plot_and_save_clusters(clusters, tsne_result, labels, file_name)

    return clusters