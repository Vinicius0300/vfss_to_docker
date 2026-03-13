
import numpy as np
import pandas as pd
import os

def get_corners_from_angle(x: float, y: float, w: float, h: float, angle_degrees: float):
    '''Get two opposite corners of a rotated rectangle based on its angle.'''
    corners = {
        'top_left': (x, y),
        'top_right': (x + w, y),
        'bottom_right': (x + w, y + h),
        'bottom_left': (x, y + h)
    }
    q1 = angle_degrees > 0 and angle_degrees < 90
    q3 = angle_degrees > -180 and angle_degrees < -90
    if q1 or q3:
        p1 = corners['top_right']
        p2 = corners['bottom_left']
    else:
        p1 = corners['top_left']
        p2 = corners['bottom_right']
    
    points = np.array([p1, p2])
    return points

def load_points(path: str, filename: str = 'Results.csv') -> np.ndarray:
        """Carrega e converte os pontos do arquivo CSV."""
        full_path = os.path.join(path,filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Arquivo de pontos não encontrado: {full_path}")

        df = pd.read_csv(full_path)
        if df.empty:
            raise ValueError(f"Arquivo de pontos vazio: {full_path}")

        row = df.iloc[0]
        return get_corners_from_angle(
            row['BX'], row['BY'], row['Width'], row['Height'], row['Angle']
        )