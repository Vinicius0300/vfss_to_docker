import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

def get_frame_from_video(video_path: str, frame_number: int):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {video_path}")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if frame_number < 1 or frame_number > total_frames:
        cap.release()
        return None # Ou raise, dependendo da sua preferência

    # Tenta o pulo rápido
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number - 1)
    
    # Verificação de segurança: O OpenCV realmente parou onde pedimos?
    current_pos = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    if current_pos != frame_number - 1:
        # Se falhou, uma alternativa é o pulo manual (lento, mas preciso)
        # cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        # for _ in range(frame_number - 1): cap.grab()
        pass 

    success, frame = cap.read()
    cap.release()

    return frame if success else None


def cria_pasta(df_data: pd.DataFrame, target: str, path_data: str, root_video_file: str, overwrite: bool = True):

    """
    Cria pastas com os frames de referência (input) e os labels de interesse (output) dos modelos

    Parâmetros:
    ------------
    df_data : pd.DataFrame
        Dataframe com os dados selecionados para o conjunto específico
    target : str
        Output esperado do modelo. "points" ou "mask"
    path_data : str
        Caminho de saída dos conjunto de input e output do modelo
    root_video_file : str
        Caminho em que encontramos os vídeos
    overwrite : bool
        Pasta, caso exista, deve ser sobrescrita
    """

    # Pasta Input -> Extrair frame de interesse
    folder_input = os.path.join(path_data, "x")
    os.makedirs(folder_input, exist_ok = overwrite)

    # Pasta Output -> Colocar mask.tif ou points.csv (copiar)
    folder_output = os.path.join(path_data, "y")
    os.makedirs(folder_output, exist_ok = overwrite)

    # Para cada linha do data frame -> extrair o frame (video_id, frame_id) e copiar file_path pro folder_output
    for i in range(df_data.shape[0]):
        df_aux = df_data.iloc[i]

        video_frame = df_aux.video_frame
        video_id = df_aux.video_id
        frame_id = df_aux.frame_id

        file_path = df_aux.target_dir
        extension = ["csv"] if target == "points" else ['tif', 'tiff']
        for file in os.listdir(file_path):
            ext = file.split(".")[-1]
            if ext in extension:
                file_path_complete = os.path.join(file_path, file)

        # Salva Frame de interesse
        video_file_path = os.path.join(root_video_file, str(video_id) + ".avi")
        frame = get_frame_from_video(video_file_path, int(frame_id))
        frame_file_path = os.path.join(folder_input, str(video_frame) + ".png")
        cv.imwrite(frame_file_path, frame)

        # Copiar Arquivo de Output (Mask ou Point)
        copy_file_output = os.path.join(folder_output, video_frame + "." + extension[0])
        shutil.copy2(file_path_complete, copy_file_output)

    print("Tamanho DataFrame: ", df_data.shape)
    print("Arquivos Input: ", len(os.listdir(folder_input)))
    print("Arquivos Output: ", len(os.listdir(folder_output)))
