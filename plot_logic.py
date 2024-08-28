from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from load_data import *
from collections import Counter
from img_transformation import get_patch_image
import seaborn as sns


def plot_mispredict_vectors(mispred_hi, mispred_flat, folder_path, annotations_path, img_folder):
    # Calculate number of errors and prediction counts for hierarchical classifiers
    for classifier_name in mispred_hi.keys():
        for feature in mispred_hi[classifier_name].keys():
            mispred_hi[classifier_name][feature]['nb_error'] = len(mispred_hi[classifier_name][feature]['predicted'])
            prediction_counts = Counter(mispred_hi[classifier_name][feature]['predicted'])
            mispred_hi[classifier_name][feature]['predicted_counts'] = prediction_counts

    # Calculate number of errors and prediction counts for flat classifiers
    for classifier_name in mispred_flat.keys():
        for feature in mispred_flat[classifier_name].keys():
            mispred_flat[classifier_name][feature]['nb_error'] = len(
                mispred_flat[classifier_name][feature]['predicted'])
            prediction_counts = Counter(mispred_flat[classifier_name][feature]['predicted'])
            mispred_flat[classifier_name][feature]['predicted_counts'] = prediction_counts

    print("Top 5 most mispredicted vectors for hierarchical classifiers:")
    for classifier_name in mispred_hi.keys():
        print(f"\nClassifier: {classifier_name}")
        sorted_features = sorted(mispred_hi[classifier_name].keys(),
                                 key=lambda x: mispred_hi[classifier_name][x]['nb_error'], reverse=True)[:5]
        for feature in sorted_features:
            img_name_pt, index, img_labels = get_image_labels_from_feature(np.array(feature), folder_path)
            print(img_name_pt)
            patch = get_patch_image(img_name_pt + '.JPG', index, annotations_path, img_folder)
            img_labels_count = Counter(img_labels)

            true_label = mispred_hi[classifier_name][feature]['true_label']
            predicted_counts = mispred_hi[classifier_name][feature]['predicted_counts']

            # Plotting
            fig, ax = plt.subplots(figsize=(11, 6))

            # Plotting the image patch
            ax.imshow(patch)
            ax.axis('off')
            ax.set_title(f"Image Patch: {img_name_pt} - True Label : {true_label} ")

            # Displaying labels and counts
            labels = list(img_labels_count.keys())
            counts = list(img_labels_count.values())
            total_count = sum(counts)
            percentages = [count / total_count * 100 for count in counts]

            label_strings = [f"{label} ({percentage:.2f}%) - {count} instances" for label, percentage, count in
                             zip(labels, percentages, counts)]
            label_text = "\n".join(label_strings)
            ax.text(1.05, 0.5, label_text, transform=ax.transAxes, fontsize=12, va='center', wrap=True)

            # Displaying predicted counts and true label
            ax.text(1.05, 0.1, f"Predicted Counts: {predicted_counts}",
                    transform=ax.transAxes, fontsize=12, va='center', wrap=True)

            plt.show()

    print("\nTop 5 most mispredicted vectors for flat classifiers:")
    for classifier_name in mispred_flat.keys():
        print(f"\nClassifier: {classifier_name}")
        sorted_features = sorted(mispred_flat[classifier_name].keys(),
                                 key=lambda x: mispred_flat[classifier_name][x]['nb_error'], reverse=True)[:5]
        for feature in sorted_features:
            img_name_pt, index, img_labels = get_image_labels_from_feature(feature, folder_path)
            patch = get_patch_image(img_name_pt + '.JPG', index, annotations_path, img_folder)
            img_labels_count = Counter(img_labels)

            true_label = mispred_flat[classifier_name][feature]['true_label']
            predicted_counts = mispred_flat[classifier_name][feature]['predicted_counts']

            # Plotting
            fig, ax = plt.subplots(figsize=(11, 6))

            # Plotting the image patch
            ax.imshow(patch)
            ax.axis('off')
            ax.set_title(f"Image Patch: {img_name_pt} - True Label : {true_label} ")

            # Displaying labels and counts
            labels = list(img_labels_count.keys())
            counts = list(img_labels_count.values())
            total_count = sum(counts)
            percentages = [count / total_count * 100 for count in counts]

            label_strings = [f"{label} ({percentage:.2f}%) - {count} instances" for label, percentage, count in
                             zip(labels, percentages, counts)]
            label_text = "\n".join(label_strings)
            ax.text(1.05, 0.5, label_text, transform=ax.transAxes, fontsize=12, va='center', wrap=True)

            # Displaying predicted counts and true label
            ax.text(1.05, 0.1, f"Predicted Counts: {predicted_counts}",
                    transform=ax.transAxes, fontsize=12, va='center', wrap=True)

            plt.show()


def plot_intermediate_metrics(flat_intermediate_reports, hi_intermediate_reports):
    # Créer une liste pour collecter les données
    rows = []

    # Itérer sur data_dict_flat
    for classifier_flat, data_flat in flat_intermediate_reports.items():
        for image_count, metrics_flat in data_flat.items():
            for label, metrics_data in metrics_flat.items():
                row = {
                    'Classifier': classifier_flat,
                    'Image Count': image_count,
                    'Label': label,
                    'Precision': metrics_data['precision'],
                    'Recall': metrics_data['recall'],
                    'F1-score': metrics_data['f1-score'],
                    'Std Precision': metrics_data['std_precision'],
                    'Std Recall': metrics_data['std_recall'],
                    'Std F1-score': metrics_data['std_f1-score']
                }
                rows.append(row)

    # Itérer sur data_dict_2
    for classifier_2, data_2 in hi_intermediate_reports.items():
        for image_count, metrics_2 in data_2.items():
            for label, metrics_data in metrics_2.items():
                row = {
                    'Classifier': classifier_2,
                    'Image Count': image_count,
                    'Label': label,
                    'Precision': metrics_data['precision'],
                    'Recall': metrics_data['recall'],
                    'F1-score': metrics_data['f1-score'],
                    'Std Precision': metrics_data['std_precision'],
                    'Std Recall': metrics_data['std_recall'],
                    'Std F1-score': metrics_data['std_f1-score']
                }
                rows.append(row)

    # Créer un DataFrame à partir des données collectées
    df = pd.DataFrame(rows)

    # Tri des classifiers uniques
    classifiers = df['Classifier'].unique()

    # Boucle sur chaque nombre d'images
    for image_count in sorted(set(df['Image Count'])):
        fig, axes = plt.subplots(3, 1, figsize=(15, 25), sharey=True)

        # Boucle sur chaque métrique
        for i, metric in enumerate(['Precision', 'Recall', 'F1-score']):
            ax = axes[i]

            # Scatter plot pour chaque classifier
            for classifier in classifiers:
                df_filtered = df[(df['Image Count'] == image_count) & (df['Classifier'] == classifier)]

                sns.scatterplot(ax=ax, data=df_filtered, x='Label', y=metric, label=classifier, markers=True)

                # Ajout des barres d'erreur
                ax.errorbar(x=df_filtered['Label'], y=df_filtered[metric],
                            yerr=df_filtered[f'Std {metric}'], fmt='o', capsize=5)

            ax.set_title(f'{metric} Comparison (Images: {image_count})')
            ax.set_xlabel('Label')
            ax.set_ylabel(metric)
            ax.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

        plt.tight_layout()
        plt.show()


def plot_coverage_comparison(df):
    # Afficher le tableau sans les colonnes Classifier et Image Count
    df_table = df.drop(columns=['Classifier', 'Image Count'])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title('Coral Coverage Comparison')
    plt.show()

    # Réorganisation des données pour le graphique
    df_melted = df.melt(id_vars=['Label', 'Classifier', 'Image Count'], var_name='Coverage Type',
                        value_name='Coverage (%)')

    # Afficher le diagramme à barres
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_melted, x='Label', y='Coverage (%)', hue='Coverage Type')

    plt.title('Coral Coverage Comparison')
    plt.xlabel('Label')
    plt.ylabel('Coverage (%)')
    plt.legend(title='Coverage Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()


def plot_metrics_by_nb_images(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports):
    plt.figure(figsize=(15, 9))

    for reports, report_type in zip([flat_reports, hi_flat_reports], ['Flat Classifier (baseline)', 'Hierarchical Classifier (ours)']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x * 25 for x in nb_images_list]

            plt.errorbar(nb_images_list, f1_scores, yerr=f1_stds, label=f'{report_type}',
                         marker='o', markersize=8, capsize=5, linewidth=4)
    plt.xlabel('Number of training patches', fontsize=36)
    plt.ylabel('F1-score'.capitalize(), fontsize=36)
    #plt.title(f'{"F1-score".capitalize()} Bottom-up vs Top-down')
    plt.legend(fontsize=34)
    plt.grid(True)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig('f1_score_plot_rio.png')
    plt.show()

    plt.figure(figsize=(15, 9))
    for reports, report_type in zip([flat_hi_reports, hi_reports], ['Flat Classifier (baseline)', 'Hierarchical Classifier (ours)']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x * 25 for x in nb_images_list]

            plt.errorbar(nb_images_list, f1_scores, yerr=f1_stds, label=f'{report_type}',
                         marker='o', markersize=8, capsize=5, linewidth=4)
    plt.xlabel('Number of training patches', fontsize=36)
    plt.ylabel('Hierarchical F1-score'.capitalize(), fontsize=36)
    #plt.title(f'{"Hierarchical F1-score".capitalize()} Bottom-up vs top-down')
    plt.legend(fontsize=34)
    plt.grid(True)

    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

    plt.tight_layout()
    plt.savefig('hierarchical_fi_score_plot_rio.png')
    plt.show()


def plot_metrics_by_nb_images_2(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports):
    # Fonction pour tracer les courbes avec des zones d'erreur colorées
    def plot_with_error_bands(ax, reports, report_type):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])

            nb_images_list = [x * 25 for x in nb_images_list]

            # Calculer les erreurs supérieures et inférieures
            f1_errors_upper = [score + std for score, std in zip(f1_scores, f1_stds)]
            f1_errors_lower = [score - std for score, std in zip(f1_scores, f1_stds)]

            # Tracer la courbe
            ax.plot(nb_images_list, f1_scores, 'o-', label=f'{classifier_name} {report_type}', markersize=8)

            # Remplir la zone d'erreur
            ax.fill_between(nb_images_list, f1_errors_lower, f1_errors_upper, alpha=0.2)

    # Création de la première figure
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(1, 1, 1)
    plot_with_error_bands(ax1, flat_reports, 'Bottom-up')
    plot_with_error_bands(ax1, hi_flat_reports, 'Top-down')
    ax1.set_xlabel('Number of training patches')
    ax1.set_ylabel('F1-score'.capitalize())
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.show()

    # Création de la deuxième figure
    plt.figure(figsize=(12, 8))
    ax2 = plt.subplot(1, 1, 1)
    plot_with_error_bands(ax2, flat_hi_reports, 'Bottom-up')
    plot_with_error_bands(ax2, hi_reports, 'Top-down')
    ax2.set_xlabel('Number of training patches')
    ax2.set_ylabel('Hierarchical F1-score'.capitalize())
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


def round_down(value, base):
    """Arrondit la valeur vers le bas au multiple le plus proche de la base."""
    return base * np.floor(value / base)


def plot_metrics_by_nb_images_twin(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports):
    fig, ax1 = plt.subplots(figsize=(15, 12))

    # Couleurs
    color_flat = 'tab:blue'
    color_hi = 'tab:orange'
    color_flat_hi = 'lightblue'
    color_hi_hi = 'lightcoral'

    # Accumulate the data
    flat_f1_scores = []
    hi_f1_scores = []
    nb_images_list = []

    for reports, report_type in zip([flat_reports, flat_hi_reports], ['Flat Classifier', 'Hierarchical Classifier']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x * 25 for x in nb_images_list]
            flat_f1_scores.extend(f1_scores)

    for reports, report_type in zip([flat_hi_reports, hi_reports], ['Flat Classifier', 'Hierarchical Classifier']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x * 25 for x in nb_images_list]
            hi_f1_scores.extend(f1_scores)

    # Find common min and max for scaling both y-axes
    min_f1 = min(flat_f1_scores + hi_f1_scores)
    max_f1 = max(flat_f1_scores + hi_f1_scores)

    # Calculate margin to extend the limits
    margin = 0.1 * (max_f1 - min_f1)  # Adjust margin percentage if needed
    extended_min = min_f1 - margin
    extended_max = max_f1 + margin

    # Round extended_min to the nearest multiple of 0.025
    rounded_min = round_down(extended_min, 0.025)

    # Plot for the flat classifier (F1-score)
    ax1.set_xlabel('Number of training patches', fontsize=25)
    ax1.set_ylabel('F1-score', color='black', fontsize=25)

    for reports, report_type in zip([flat_reports, hi_flat_reports], ['Flat Classifier', 'Hierarchical Classifier']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x for x in nb_images_list]

            if report_type == 'Flat Classifier':
                color = color_flat
            else:
                color = color_hi

            ax1.errorbar(nb_images_list, f1_scores, yerr=f1_stds, label=f'{classifier_name} {report_type}',
                         marker='o', markersize=8, capsize=5, color=color)

    # Set y-axis limits and ticks
    ax1.set_ylim(rounded_min, extended_max)
    ax1.set_yticks(np.arange(rounded_min, extended_max + 0.025, 0.025))  # Set fixed interval for y-ticks

    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.grid(True)

    # Create a second y-axis to plot the hierarchical classifier (F1-score)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Hierarchical F1-score', color='grey', fontsize=25)

    for reports, report_type in zip([flat_hi_reports, hi_reports], ['Flat Classifier', 'Hierarchical Classifier']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x for x in nb_images_list]

            if report_type == 'Flat Classifier':
                color = color_flat_hi
            else:
                color = color_hi_hi

            ax2.errorbar(nb_images_list, f1_scores, yerr=f1_stds, label=f'{classifier_name} {report_type}',
                         marker='o', markersize=8, capsize=5, linestyle='--', color=color)

    # Set y-axis limits and ticks for the second y-axis
    ax2.set_ylim(rounded_min, extended_max)
    ax2.set_yticks(np.arange(rounded_min, extended_max + 0.025, 0.025))  # Set fixed interval for y-ticks

    ax2.tick_params(axis='y', labelcolor='grey', labelsize=20)
    #ax2.legend(loc='upper right', fontsize=25)
    plt.tight_layout()
    # Save the plot if needed
    # plt.savefig('combined_f1_score_plot.png')
    plt.show()


def plot_legend_separately():
    # Couleurs
    color_flat = 'tab:blue'
    color_hi = 'tab:orange'
    color_flat_hi = 'lightblue'
    color_hi_hi = 'lightcoral'

    # Création d'un nouveau graphique pour la légende
    fig, ax = plt.subplots(figsize=(10, 2))

    # Tracé de lignes pour la légende avec les labels donnés
    ax.plot([], [], color=color_flat, marker='o', markersize=8, linestyle='-', label='Flat Classifier MLP')
    ax.plot([], [], color=color_hi, marker='o', markersize=8, linestyle='-', label='Hierarchical Classifier Per Parent Node')
    ax.plot([], [], color=color_flat_hi, marker='o', markersize=8, linestyle='--', label='Flat Classifier MLP')
    ax.plot([], [], color=color_hi_hi, marker='o', markersize=8, linestyle='--', label='Hierarchical Classifier Per Parent Node')

    # Création de la légende
    leg = ax.legend(loc='center', fontsize=15, ncol=2, frameon=False)

    # Modifier la couleur des étiquettes pour les lignes en pointillés
    for text, line in zip(leg.get_texts(), ax.get_lines()):
        if line.get_linestyle() == '--':
            text.set_color('grey')

    # Ajouter les textes pour "F1 score" et "Hierarchical F1 score"
    fig.text(0.1, 0.8, "F1 score", color='black', fontsize=18)
    fig.text(0.6, 0.8, "Hierarchical F1 score", color='grey', fontsize=18)

    # Suppression des axes
    ax.axis('off')

    # Sauvegarder l'image de la légende
    plt.savefig('legend_only.png', bbox_inches='tight')
    plt.show()


def plot_patches(folder_path, img_folder, annotations_path, labels):
    """
    labels = ['PAL', 'TFL', 'SINV_BLC', 'ZOS']
folder_path = '.features/efficientnet-b0_features'
img_folder = './Images-2'
annotations_path = 'annotations/annotations.csv'

plot_patches(folder_path, img_folder, annotations_path, labels)
    :param folder_path:
    :param img_folder:
    :param annotations_path:
    :param labels:
    :return:
    """
    # Préparation de la figure
    fig, axs = plt.subplots(len(labels), 5, figsize=(15, 3 * len(labels)))

    for row, label in enumerate(labels):
        features_labels = get_features_from_label_type(folder_path, label)
        features_labels_random = random.sample(features_labels, 5)

        for col, feature in enumerate(features_labels_random):
            img_name, index, _ = get_image_labels_from_feature(feature, folder_path)
            print(img_name, index)
            img_name = img_name + '.JPG'
            patch = get_patch_image(img_name, index, annotations_path, img_folder)

            if len(labels) == 1:
                ax = axs[col]
            else:
                ax = axs[row, col]

            ax.imshow(patch)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, rotation=90, size='large')

    for i, label in enumerate(labels):
        axs[i, 0].annotate(label, xy=(-0.2, 0.5), xycoords='axes fraction',
                           ha='center', va='center', size='large')

    plt.tight_layout()
    plt.show()
