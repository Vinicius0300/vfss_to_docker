import pandas as pd
import os
import numpy as np
import random
import torch
import math
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import operator
import seaborn as sns

from src.data_extraction.labels import get_label_metadata
from src.data_extraction.attributions import get_attributions_metadata
from src.data_extraction.video_frame import create_video_frame_metadata_from_label_and_attributions
from src.create_folders import get_frame_from_video
from src.target.points import load_points
from src.data_comparison import *



# Verificando resoluções das Máscaras
def check_mask_resolution(video_frame, label_folder, video_folder, video_id, frame_id):
    path_label = os.path.join(label_folder, "Mask.tif")
    path_video = os.path.join(video_folder, f"{video_id}.avi")
    
    frame = get_frame_from_video(path_video, int(frame_id))
    mask = cv2.imread(path_label)
    
    if frame is None:
        raise Exception("Frame não encontrado\n\t"
                        f"Frame: {video_frame}\n\t" \
                        f"Video Path: {path_video}")
    if mask is None:
        raise Exception("Máscara não encontrada\n\t"
                        f"Frame: {video_frame}\n\t" \
                        f"Label Path: {path_label}")
    if frame.shape != mask.shape:
        print(f"\nResolução de Máscara Incorreta\n\t" \
              f"Frame: {video_frame}\n\t" \
              f"Frame Original: {frame.shape}\n\t" \
              f"Máscara Rotulada: {mask.shape}\n\t" \
              f"Label Folder: {label_folder}")

    return

# Verificando Posições dos Pontos
def check_point_position(video_frame, label_folder, video_folder, video_id, frame_id):
    path_video = os.path.join(video_folder, f"{video_id}.avi")

    frame = get_frame_from_video(path_video, int(frame_id))
    points = load_points(label_folder, "Results.csv")
    h, w, _ = frame.shape
    
    for point in points:
        if h <= point[0] or w <= point[1]:
            print(f"\nPosição de Ponto Inválida\n\t" \
                  f"Frame: {video_frame}\n\t" \
                  f"Frame Original: {frame.shape}\n\t" \
                  f"Pontos Marcados: {point}\n\t" \
                  f"Label Folder: {label_folder}")

    return

# Aplicando verificações iniciais
def verifica_dimensoes(row, video_folder):
    has_mask = row["has_mask"]
    has_points = row["has_points"]
    video_frame = row["video_frame"]
    video_id = row["video_id"]
    frame_id = row["frame_id"]
    file_path = row["file_path"]

    if has_mask:
        check_mask_resolution(video_frame, file_path, video_folder, video_id, frame_id)
    if has_points:
        check_point_position(video_frame, file_path, video_folder, video_id, frame_id)


# Inverte coloração da Máscara
def invert_color_mask(mask):
    mask_invert = cv2.bitwise_not(mask)
    return mask_invert

# Checa Quais Máscaras Estão Invertidas
def check_inverted_masks(row):
    has_mask = row["has_mask"]
    video_frame = row["video_frame"]
    file_path = row["file_path"]

    if has_mask:
        path_label = os.path.join(file_path, "Mask.tif")
        mask = cv2.imread(path_label)
        moda = stats.mode(mask, axis=None)
        if moda.mode == 255:
            print(f"\nMáscara Invertida\n\t" \
                  f"Frame: {video_frame}\n\t" \
                  f"Label Folder: {file_path}\n\t" \
                  f"Moda: {moda.mode}")
            mask_invert = invert_color_mask(mask)
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(mask, cmap='gray')
            ax[0].axis('off')
            ax[1].imshow(mask_invert, cmap = "gray")
            ax[1].axis('off')
            plt.show()
        elif moda.mode != 0:
            print(f"\nModa da máscara Incoerente (diferente de 0 ou 1)\n\t" \
                  f"Frame: {video_frame}\n\t" \
                  f"Label Folder: {file_path}\n\t" \
                  f"Moda: {moda.mode}")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.show()
    return

# Corrige e Salva Quais Máscaras Estão Invertidas
def correct_inverted_masks(row):
    has_mask = row["has_mask"]
    video_frame = row["video_frame"]
    file_path = row["file_path"]

    if has_mask:
        path_label = os.path.join(file_path, "Mask.tif")
        mask = cv2.imread(path_label)
        moda = stats.mode(mask, axis=None)
        if moda.mode == 255:
            mask_invert = invert_color_mask(mask)
            cv2.imwrite(path_label, mask_invert)
            print(f"\nMáscara Invertida\n\t" \
                  f"Frame: {video_frame}\n\t" \
                  f"Label Folder: {file_path}\n\t" \
                  f"Moda: {moda.mode}")
        elif moda.mode != 0:
            print(f"\nModa da máscara Incoerente (diferente de 0 ou 1)\n\t" \
                  f"Frame: {video_frame}\n\t" \
                  f"Label Folder: {file_path}\n\t" \
                  f"Moda: {moda.mode}")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.show()
    return


# Gerando Cores para gráficos
def generate_color_list(n, cmap_name="spring"):
    cmap = plt.get_cmap(cmap_name)

    # tab10 possui 10 cores bem distintas
    if hasattr(cmap, "colors"):
        colors = [cmap.colors[i % len(cmap.colors)] for i in range(n)]
    else:
        colors = [cmap(i) for i in np.linspace(0, 1, n)]

    return colors

# Visualizando Marcações de Pontos
def plot_coherence_points(df_video_frame, video_folder, output_folder = ""):
    n_rotuladores = df_video_frame.shape[0]
    colors = generate_color_list(n_rotuladores)

    video_id = df_video_frame["video_id"].values[0]
    frame_id = df_video_frame["frame_id"].values[0]
    video_frame = df_video_frame["video_frame"].values[0]

    path_video = os.path.join(video_folder, f"{video_id}.avi")
    frame = get_frame_from_video(path_video, int(frame_id))
    
    fig, ax = plt.subplots(1,n_rotuladores+1, figsize=(5*(n_rotuladores+1),5))
    
    # Plot por Rotulador
    all_points = []
    all_labelers = []
    all_distance = []
    for i in range(n_rotuladores):
        df_rotulador = df_video_frame.loc[i]
        labeler = df_rotulador['labeler']
        ax[i].imshow(frame)
        if df_rotulador["has_points"]:
            points = load_points(df_rotulador["file_path"], "Results.csv")
            distance = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
            ponto_medio = (points[0] + points[1]) / 2

            ax[i].scatter(points[:, 0], points[:, 1], color=colors[i], marker='o', s=5, linewidths=1)
            ax[i].plot(points[:, 0], points[:, 1], color=colors[i], lw=1, alpha = 0.3)
            ax[i].text(ponto_medio[0] + 10, ponto_medio[1], f'{distance:.1f} px', 
                       color=colors[i], fontweight='bold', fontsize=12)
            all_points.append(points)
            all_labelers.append(labeler)
            all_distance.append(distance)
        else:
            ax[i].text(frame.shape[1]/2, frame.shape[0]/2, 'NÃO ROTULADO', 
                       color='red', fontweight='bold', fontsize=20, ha='center', va='center')
        ax[i].axis("off")
        ax[i].set_title(f"{labeler} - {video_frame}")

    # Plot Geral 
    ax[n_rotuladores].imshow(frame)
    for i in range(len(all_points)):
        points = all_points[i]
        ax[n_rotuladores].scatter(points[:, 0], points[:, 1], color=colors[i],
                                  marker='o', s=5, linewidths=1, label = f"{all_labelers[i]} - ({all_distance[i]:.1f} px)")
        ax[n_rotuladores].plot(points[:, 0], points[:, 1], color=colors[i], lw=1, alpha = 0.3)
    ax[n_rotuladores].axis("off")
    ax[n_rotuladores].set_title("Sobreposição de Pontos")
    ax[n_rotuladores].legend()
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, video_frame + ".png"))
    plt.show()
    
    return 

# Visualização Marcações de Máscaras
def plot_coherence_masks(df_video_frame, video_folder, output_dir = "", alpha=0.4):
    n_rotuladores = df_video_frame.shape[0]
    colors = generate_color_list(n_rotuladores)

    video_id = df_video_frame["video_id"].values[0]
    frame_id = df_video_frame["frame_id"].values[0]
    video_frame = df_video_frame["video_frame"].values[0]

    path_video = os.path.join(video_folder, f"{video_id}.avi")
    frame = get_frame_from_video(path_video, int(frame_id))
    
    fig, ax = plt.subplots(1, n_rotuladores + 1, figsize=(5 * (n_rotuladores + 1), 5))
    
    all_masks = []
    all_labelers = []

    for i in range(n_rotuladores):
        df_rotulador = df_video_frame.iloc[i]
        labeler = df_rotulador['labeler']
        
        ax[i].imshow(frame, cmap='gray')
        
        if df_rotulador["has_mask"]:

            path_label = os.path.join(df_rotulador["file_path"], "Mask.tif")
            mask = cv2.imread(path_label, cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            if mask.shape[:2] != frame.shape[:2]:
                ax[i].text(frame.shape[1]/2, frame.shape[0]/2, 'FORMATO\nINCORRETO',
                        color='red', fontweight='bold', fontsize=20, ha='center', va='center')
            else:
                # Criando a sobreposição colorida
                mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
                rgb = plt.cm.colors.to_rgb(colors[i])
                mask_rgba[..., :3] = rgb
                mask_rgba[..., 3] = (mask > 0) * alpha

                ax[i].imshow(mask_rgba)
                all_masks.append(mask)
                all_labelers.append(labeler)
        else:
            ax[i].text(frame.shape[1]/2, frame.shape[0]/2, 'SEM MÁSCARA', 
                       color='red', fontweight='bold', fontsize=20, ha='center', va='center')
        
        ax[i].axis("off")
        ax[i].set_title(f"{labeler} - {video_frame}")

    # Plot Geral
    ax[n_rotuladores].imshow(frame, cmap='gray')
    for i, mask in enumerate(all_masks):
        mask_rgba = np.zeros((*mask.shape, 4))
        rgb = plt.cm.colors.to_rgb(colors[i])
        mask_rgba[..., :3] = rgb
        mask_rgba[..., 3] = (mask > 0) * alpha
        ax[n_rotuladores].imshow(mask_rgba, label=all_labelers[i])
        
    ax[n_rotuladores].axis("off")
    ax[n_rotuladores].set_title("Sobreposição de Máscaras")
    
    if output_dir:
        plt.savefig(os.path.join(output_dir,video_frame + ".png"))
    plt.show()


# Métricas para Pontos em um mesmo frame
def metric_coherence_points(df, folder_video):

    rows = []

    video_frame_unique = df["video_frame"].unique()
    for video_frame in video_frame_unique:
        df_video_frame = df.loc[df["video_frame"] == video_frame].reset_index()
        n_labels = df_video_frame["has_points"].values.sum()

        if n_labels == 2:
            row = {}
            path_video = os.path.join(folder_video, f'{df_video_frame["video_id"].values[0]}.avi')
            frame = get_frame_from_video(path_video, int(df_video_frame["frame_id"].values[0]))
            h, w, _ = frame.shape
            pC2_1, pC4_1 = load_points(df_video_frame["file_path"].values[0], "Results.csv")
            pC2_2, pC4_2 = load_points(df_video_frame["file_path"].values[1], "Results.csv")
            row["distance_C2"] = np.linalg.norm(pC2_1 - pC2_2)/h
            row["distance_C4"] = np.linalg.norm(pC4_1 - pC4_2)/h
            row["distance_points"] = (np.linalg.norm(pC2_1 - pC2_2) + np.linalg.norm(pC4_1 - pC4_2))/h
            row["difference_length"] = np.abs(np.linalg.norm(pC2_1 - pC4_1) - np.linalg.norm(pC2_2 - pC4_2))/h
            row["labeler_1"] = df_video_frame["labeler"].values[0]
            row["labeler_2"] = df_video_frame["labeler"].values[1]
            row["video_frame"] = video_frame
            rows.append(row)
        
        elif n_labels > 2:
            print(f"Número de Rotuladores maior que dois para {video_frame}")
    
    return pd.DataFrame(rows)

# Métricas para Máscaras em um mesmo frame
def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    return (2. * intersection) / (mask1.sum() + mask2.sum() + 1e-8)

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)

def metric_coherence_masks(df):

    rows = []
    video_frame_unique = df["video_frame"].unique()

    for video_frame in video_frame_unique:
        df_video_frame = df.loc[df["video_frame"] == video_frame].reset_index()
        n_labels = df_video_frame["has_mask"].values.sum()

        if n_labels == 2:
            

            # carregar máscaras
            mask1 = cv2.imread(os.path.join(df_video_frame["file_path"].values[0], "Mask.tif"))
            mask2 = cv2.imread(os.path.join(df_video_frame["file_path"].values[1], "Mask.tif"))
            mask1 = (mask1 > 0).astype(np.uint8)
            mask2 = (mask2 > 0).astype(np.uint8)
            if mask1.shape[0] == mask2.shape[0]:
                row = {}
                # métricas principais
                row["dice"] = dice_coefficient(mask1, mask2)
                row["iou"] = iou(mask1, mask2)

                # métricas extras (opcional, mas bom pra paper)
                area1 = int(mask1.sum())
                area2 = int(mask2.sum())

                row["area_diff"] = abs(area1 - area2) / (area1 + 1e-8)

                # metadados
                row["labeler_1"] = df_video_frame["labeler"].values[0]
                row["labeler_2"] = df_video_frame["labeler"].values[1]
                row["video_frame"] = video_frame
                rows.append(row)
        elif n_labels > 2:
            print(f"Número de Rotuladores maior que dois para {video_frame}")
    return pd.DataFrame(rows)



# Plotando Anotações
def plot_comparison(df, cols):
    fig, ax = plt.subplots(len(cols),1, figsize = (15,9))
    for i in range(len(cols)):
        ax[i].hist(df[cols[i]], bins = 100, zorder = 2)
        ax[i].set_title(cols[i])
        ax[i].grid(alpha = 0.4, zorder = 1)
        ax[i].set_ylim(0,50)
        ax[i].set_xticks(np.arange(0, (df[cols[i]].max())*1.05, (df[cols[i]].max())*1.05/30)) 
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
    plt.tight_layout()
    plt.show()

def get_annotations(dfDados, dfMetric, col_metric, videos_dir, threshold, op=operator.ge, dtype="points"):
    dfBad = dfMetric.loc[op(dfMetric[col_metric], threshold)]
    
    for video_frame in dfBad["video_frame"]:
        dfDadosBad = dfDados.loc[dfDados["video_frame"] == video_frame].reset_index()
        
        if dtype == "points":
            plot_coherence_points(dfDadosBad, videos_dir)
        else:
            plot_coherence_masks(dfDadosBad, videos_dir)


# Plot Matriz de Metrica
def plot_metric_matrix(dfMetric, list_col):
    # Invariante a ordem dos rotuladores
    dfMetric[["a1", "a2"]] = np.sort(dfMetric[["labeler_1", "labeler_2"]], axis=1)
    
    fig, ax = plt.subplots(1, len(list_col), figsize = (len(list_col)*6, 5))
    for i, col in enumerate(list_col):
        df_mean = (
            dfMetric.groupby(["a1", "a2"])[col]
            .mean()
            .reset_index()
        )
        matrix = df_mean.pivot(index="a1", columns="a2", values=col)
        matrix = matrix.combine_first(matrix.T)
        np.fill_diagonal(matrix.values, np.nan)
        sns.heatmap(matrix, annot=True, cmap="viridis", vmin=matrix.min().min(), vmax=matrix.max().max(),  ax=ax[i])
        ax[i].set_title(f"Concordância Média - {col}")
    plt.show()
