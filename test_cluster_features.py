from hierarchical_utils import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


def cluster_features(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters, kmeans


def analyze_clusters(clusters, labels, n_clusters=5):
    labels = np.array(labels)
    cluster_analysis = []
    for cluster_num in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_num)[0]
        cluster_labels = labels[cluster_indices]
        label_counts = Counter(cluster_labels)
        cluster_analysis.append(label_counts)
    return cluster_analysis


if __name__ == '__main__':
    folder_path = '.features/efficientnet-b0_features'
    image_names, features, labels = get_all_features_img_name(folder_path)
    print(len(set(labels)))
    n_clusters = len(set(labels))
    clusters, kmeans = cluster_features(features, n_clusters)
    cluster_analysis = analyze_clusters(clusters, labels, n_clusters)

    print(cluster_analysis)

    df = pd.DataFrame(cluster_analysis)
    df = df.fillna(0)  # Remplacer les valeurs NaN par 0

    num_colors = 54
    palette = sns.color_palette("husl", num_colors)
    cmap = ListedColormap(palette)

    # Création du graphique en barres empilées avec la palette personnalisée
    ax = df.plot(kind='bar', stacked=True, figsize=(15, 10), colormap=cmap)
    for p in ax.patches:
        if p.get_width() > 0:  # Vérifier que la barre est visible
            x = p.get_x() + p.get_width() / 2  # Position x du label au milieu de la barre
            y = p.get_y() + p.get_height() / 2  # Position y du label au milieu de la barre
            value = int(p.get_height())  # Valeur à afficher
            ax.annotate(f'{value}', (x, y), ha='center', va='center', fontsize=8, color='black', weight='bold')

    # Ajouter des labels et des titres
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution des labels par cluster')

    # Afficher la légende
    #plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Afficher le graphique
    plt.tight_layout()
    plt.show()

    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(df)

    # Générer une palette de couleurs avec suffisamment de couleurs pour tous les clusters
    num_clusters = len(df)
    colors = cm.get_cmap('tab20b', num_clusters)

    # Création du graphique 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ajouter les points au graphique 3D
    sc = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=np.arange(num_clusters),
                    cmap=colors)

    # Ajouter des labels et des titres
    ax.set_title('Clusters Visualisés avec PCA 3D')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    # Ajouter une légende
    plt.colorbar(sc, ax=ax, label='Cluster')

    # Afficher le graphique
    plt.show()


