"""
label_editor.py
===============
Aplicação interativa (napari) para re-rotulagem de frames de vídeos médicos (.avi).

Integra-se com o pipeline de revisão existente recebendo dois DataFrames:
  - review_df           : saída do pipeline de revisão (video_frame, rotulo, motivo)
  - video_frame_metadata_df : metadados de cada frame (file_path, has_mask, has_points, ...)

Uso
---
    from label_editor import run_label_editor

    run_label_editor(
        review_csv_path          = "revisao_rotulador1.csv",
        video_frame_metadata_df  = metadata_df,
        videos_root              = "/data/videos",        # pasta com os .avi
        output_root              = "/data/revisados",     # pasta de saída (nova)
        video_filename_pattern   = "video_{video_id}.avi" # padrão do nome do arquivo
    )

Dependências
------------
    pip install napari opencv-python-headless pandas numpy tifffile
    # napari requer PyQt5 ou PySide2 para o backend de GUI:
    pip install "napari[pyqt5]"

Funções externas esperadas (imports de responsabilidade do chamador)
--------------------------------------------------------------------
    from src.target.points import load_points

    load_points(file_path, filename="Results.csv") -> np.ndarray
        Carrega o Results.csv e retorna array (2,2) [[x1,y1],[x2,y2]].

    get_corners_from_angle(x, y, w, h, angle) -> np.ndarray
        Retorna dois cantos opostos do bounding box rotacionado.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import napari
import numpy as np
from src.target.points import load_points
try:
    from IPython import get_ipython as _get_ipython
except ImportError:
    _get_ipython = None
import pandas as pd
import tifffile
from napari.qt.threading import create_worker
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSlider,
)
from qtpy.QtCore import Qt


# ---------------------------------------------------------------------------
# Helpers do pipeline existente
# ---------------------------------------------------------------------------

def points_to_bbox(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float, float, float]:
    """Converte dois pontos editados em (BX, BY, Width, Height, Angle).

    O ângulo reconstruído é um valor proxy que preserva o mesmo quadrante
    do ângulo original, garantindo round-trip perfeito com get_corners_from_angle:
    o par de cantos escolhido (top_right/bottom_left vs top_left/bottom_right)
    depende apenas do quadrante, não do valor exato do ângulo.

    Quadrantes (convenção ImageJ):
      - top_right / bottom_left  → Q1 (0..90)  ou Q3 (-180..-90) → proxy = -135°
      - top_left  / bottom_right → demais casos                   → proxy =  -45°
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    import math
    # Passo 1: arredondar coordenadas dos pontos para inteiros (floor)
    # Todos os cálculos derivados partem desses valores inteiros
    x1 = math.floor(x1)
    y1 = math.floor(y1)
    x2 = math.floor(x2)
    y2 = math.floor(y2)

    bx = int(min(x1, x2))
    by = int(min(y1, y2))
    w  = int(abs(x2 - x1))
    h  = int(abs(y2 - y1))

    # Descobre qual ponto está no topo (menor y)
    top_x = x1 if y1 <= y2 else x2

    # Se o ponto do topo é o da direita (x == bx+w) → par top_right/bottom_left → Q3
    if top_x == bx + w:
        angle = -135.0   # proxy Q3: qualquer valor em (-180, -90) funciona
    else:
        angle = -45.0    # proxy Q2/Q4: qualquer valor fora de Q1 e Q3 funciona

    # Angle com 3 casas decimais — calculado de coordenadas inteiras
    angle = round(angle, 3)

    return bx, by, w, h, angle


def get_frame_from_video(path_video: str, frame_id: int) -> np.ndarray:
    """Extrai um frame de um vídeo .avi.

    Retorna array HxWxC (BGR) ou HxW (grayscale).
    """
    cap = cv2.VideoCapture(path_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Não foi possível ler frame {frame_id} de {path_video}")
    return frame  # BGR


# ---------------------------------------------------------------------------
# Funções de I/O de saída
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_points(
    points: np.ndarray,
    output_dir: str,
    original_csv_path: str,
    filename: str = "Results.csv",
) -> None:
    """Reconstrói e salva o Results.csv no formato original do ImageJ/Fiji.

    Parâmetros
    ----------
    points : np.ndarray, shape (2, 2)
        Dois pontos editados em coordenadas de imagem [[x1,y1],[x2,y2]].
    output_dir : str
        Pasta de destino.
    original_csv_path : str
        Caminho para o Results.csv original — preserva colunas secundárias
        (Area, Min, Max, Circ., etc.) e o índice da linha tal como o ImageJ gerou.
    filename : str
        Nome do arquivo de saída. Padrão: "Results.csv".
    """
    _ensure_dir(output_dir)

    bx, by, w, h, angle = points_to_bbox(points[0], points[1])

    if original_csv_path is not None and os.path.exists(original_csv_path):
        # Preserva todas as colunas secundárias do CSV original (Area, Min, Max, etc.)
        original_df = pd.read_csv(original_csv_path)
        original_row = original_df.iloc[0].copy()
        original_row["BX"]     = bx      # int, vem de points_to_bbox
        original_row["BY"]     = by      # int
        original_row["Width"]  = w       # int
        original_row["Height"] = h       # int
        original_row["Angle"]  = angle   # float, 3 casas decimais
        if "Length" in original_row and not pd.isna(original_row["Length"]):
            original_row["Length"] = round(float(original_row["Length"]), 3)
        new_df = pd.DataFrame([original_row])
    else:
        # Criação do zero — gera apenas as colunas essenciais usadas pelo pipeline
        # Colunas secundárias preenchidas com NaN (artefatos ImageJ não disponíveis)
        new_df = pd.DataFrame([{
            "Area": np.nan, "Min": np.nan, "Max": np.nan,
            "BX": bx, "BY": by, "Width": w, "Height": h,
            "Angle": angle,
            "Circ.": np.nan, "Slice": np.nan, "AR": np.nan,
            "Round": np.nan, "Solidity": np.nan, "Length": np.nan,
        }])

    # Salva mantendo o índice original (primeira coluna sem nome, valor "1")
    out_path = str(Path(output_dir) / filename)
    new_df.to_csv(out_path, index=True, index_label="")
    print(f"    BX={bx:.1f}  BY={by:.1f}  W={w:.1f}  H={h:.1f}  Angle={angle:.3f} deg")


def save_mask(
    mask: np.ndarray,
    output_dir: str,
    filename: str = "Mask.tif",
    target_shape: tuple | None = None,
) -> None:
    """Salva máscara binária como Mask.tif no output_dir.

    Se target_shape=(H, W) for fornecido e diferir do shape da máscara,
    redimensiona com interpolação nearest (preserva valores binários 0/1)
    para garantir que a máscara tenha a mesma resolução do frame do vídeo.
    """
    _ensure_dir(output_dir)
    # Garante array 2D uint8 com valores 0 e 1
    out = np.asarray(mask, dtype=np.uint8)
    if out.ndim == 3:
        out = out[..., 0]   # remove canal extra se vier 3D
    out = np.where(out > 0, 255, 0).astype(np.uint8)  # binário 0/255 — visível em qualquer visualizador

    if target_shape is not None:
        th, tw = int(target_shape[0]), int(target_shape[1])
        if out.shape != (th, tw):
            out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_NEAREST)

    out_path = str(Path(output_dir) / filename)
    # cv2.imwrite garante uint8 com valores 0/255 preservados corretamente no TIFF
    cv2.imwrite(out_path, out)

    # Verifica imediatamente se o arquivo foi salvo corretamente
    check = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    if check is not None:
        print(f"    Verificação pós-save: {int((check > 0).sum())} pixels não-zero no arquivo")
        print(f"    Valores únicos no arquivo: {np.unique(check).tolist()}")
    else:
        print(f"    [AVISO] Não foi possível verificar o arquivo salvo: {out_path}")


# ---------------------------------------------------------------------------
# Widget de controle lateral
# ---------------------------------------------------------------------------

class LabelEditorControls(QWidget):
    """Painel lateral com informações do frame atual e botões de navegação."""

    def __init__(self, editor: "LabelEditorApp"):
        super().__init__()
        self.editor = editor
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # --- Info do frame atual ---
        self.lbl_progress = QLabel()
        self.lbl_progress.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.lbl_progress)

        self.lbl_video_frame = QLabel()
        layout.addWidget(self.lbl_video_frame)

        self.lbl_rotulo = QLabel()
        layout.addWidget(self.lbl_rotulo)

        self.lbl_motivo = QLabel()
        self.lbl_motivo.setWordWrap(True)
        layout.addWidget(self.lbl_motivo)

        self.lbl_status = QLabel()
        self.lbl_status.setStyleSheet("color: #888; font-style: italic;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addSpacing(10)

        # --- Instrucoes dinamicas ---
        self.lbl_instrucoes = QLabel()
        self.lbl_instrucoes.setWordWrap(True)
        self.lbl_instrucoes.setStyleSheet(
            "background: #2a2a2a; padding: 8px; border-radius: 4px; font-size: 11px;"
        )
        layout.addWidget(self.lbl_instrucoes)

        layout.addSpacing(10)

        # --- Controle de tamanho do pincel (mask) ---
        self.brush_widget = QWidget()
        brush_layout = QVBoxLayout()
        brush_layout.setContentsMargins(0, 0, 0, 0)
        brush_label = QLabel("Tamanho do pincel / borracha:")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        self.brush_size_label = QLabel("10")
        brush_layout.addWidget(brush_label)
        brush_layout.addWidget(self.brush_slider)
        brush_layout.addWidget(self.brush_size_label)
        self.brush_widget.setLayout(brush_layout)
        layout.addWidget(self.brush_widget)

        layout.addSpacing(10)

        # --- Botões principais ---
        btn_salvar = QPushButton("💾  Salvar e Avançar")
        btn_salvar.setStyleSheet(
            "background-color: #2d7a2d; color: white; font-weight: bold; padding: 6px;"
        )
        btn_salvar.clicked.connect(self.editor.save_and_next)
        layout.addWidget(btn_salvar)

        btn_pular = QPushButton("⏭  Pular (sem salvar)")
        btn_pular.clicked.connect(self.editor.skip_frame)
        layout.addWidget(btn_pular)

        layout.addSpacing(4)

        nav_layout = QHBoxLayout()
        btn_prev = QPushButton("◀ Anterior")
        btn_prev.clicked.connect(self.editor.go_previous)
        btn_next = QPushButton("Próximo ▶")
        btn_next.clicked.connect(self.editor.go_next)
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        layout.addLayout(nav_layout)

        layout.addStretch()

        # --- Rodapé ---
        lbl_footer = QLabel("label_editor · pipeline de revisão")
        lbl_footer.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(lbl_footer)

        self.setLayout(layout)
        self.setMinimumWidth(240)

    def _on_brush_size_changed(self, value: int):
        self.brush_size_label.setText(str(value))
        self.editor.set_brush_size(value)

    def update_info(
        self,
        index: int,
        total: int,
        video_frame: str,
        rotulo: str,
        motivo: str,
        revisado: bool,
        has_existing: bool,
    ):
        self.lbl_progress.setText(f"Frame {index + 1} / {total}")
        self.lbl_video_frame.setText(f"📹 {video_frame}")
        self.lbl_rotulo.setText(f"🏷  Rótulo: <b>{rotulo}</b>")
        self.lbl_motivo.setText(f"⚠  Motivo: {motivo}")
        status = "✅ Já revisado" if revisado else "🔄 Pendente"
        self.lbl_status.setText(status)

        # Instruções específicas por tipo
        if rotulo == "point":
            modo = "editar pontos existentes" if has_existing else "criar pontos do zero"
            instrucoes = (
                f"<b>Modo: {modo}</b><br><br>"
                "• Use a camada <i>Points</i> no napari.<br>"
                "• <b>Adicionar ponto:</b> tecle <code>P</code> e clique.<br>"
                "• <b>Selecionar/mover:</b> tecle <code>S</code>, clique no ponto "
                "e clique no novo local.<br>"
                "• <b>Apagar ponto:</b> selecione e pressione <code>Delete</code>.<br>"
                "• Mantenha <b>exatamente 2 pontos</b> (C2 e C4) antes de salvar.<br>"
                "• Ordem: primeiro ponto = C2, segundo = C4."
            )
        else:
            modo = "editar máscara existente" if has_existing else "criar máscara do zero"
            instrucoes = (
                f"<b>Modo: {modo}</b><br><br>"
                "• Use a camada <i>Mask</i> no napari.<br>"
                "• <b>Pintar:</b> ferramenta <i>paint brush</i> (ícone pincel).<br>"
                "• <b>Borracha:</b> ferramenta <i>erase</i> (ícone borracha).<br>"
                "• Ajuste o tamanho do pincel/borracha no controle acima.<br>"
                "• Valores <b>1 = máscara</b>, <b>0 = fundo</b>."
            )

        self.lbl_instrucoes.setText(instrucoes)

        # Mostrar/ocultar controle de pincel
        self.brush_widget.setVisible(rotulo == "mask")


# ---------------------------------------------------------------------------
# Aplicação principal
# ---------------------------------------------------------------------------

class LabelEditorApp:
    """
    Gerencia o loop de edição frame a frame usando napari.

    Parâmetros
    ----------
    review_csv_path : str
        Caminho para o CSV do rotulador (video_frame, rotulo, motivo).
    video_frame_metadata_df : pd.DataFrame
        Metadados de cada frame: colunas obrigatórias são
        ``video_frame``, ``file_path``, ``has_mask``, ``has_points``.
    videos_root : str
        Pasta raiz contendo os arquivos .avi.
    output_root : str or None
        Pasta de saída onde os rótulos editados serão salvos.
        Se None, sobrescreve os arquivos originais em file_path.
    video_filename_pattern : str
        Padrão do nome do arquivo de vídeo. Use ``{video_id}`` como
        placeholder. Ex.: ``"video_{video_id}.avi"``.
    """

    def __init__(
        self,
        review_csv_path: str,
        video_frame_metadata_df: pd.DataFrame,
        videos_root: str,
        output_root: str | None,
        labeler: str,
        video_filename_pattern: str = "video_{video_id}.avi",
    ):
        self.review_csv_path = review_csv_path
        self.metadata_df = video_frame_metadata_df.copy()
        self.videos_root = videos_root
        self.output_root = output_root
        self.labeler = labeler
        self.video_filename_pattern = video_filename_pattern

        # Carrega o CSV de revisão
        self.review_df = pd.read_csv(review_csv_path)
        self._validate_review_df()

        # Garante coluna 'revisado'
        if "revisado" not in self.review_df.columns:
            self.review_df["revisado"] = False
        if "revisado_em" not in self.review_df.columns:
            self.review_df["revisado_em"] = pd.NaT

        self.current_index: int = 0
        self.viewer: Optional[napari.Viewer] = None
        self.controls: Optional[LabelEditorControls] = None

        # Layers ativas
        self._img_layer = None
        self._points_layer = None
        self._mask_layer = None

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------

    def _validate_review_df(self):
        required = {"video_frame", "rotulo", "motivo"}
        missing = required - set(self.review_df.columns)
        if missing:
            raise ValueError(f"review_df faltam colunas: {missing}")
        valid_rotulos = {"point", "mask"}
        invalid = set(self.review_df["rotulo"].unique()) - valid_rotulos
        if invalid:
            raise ValueError(f"Valores inválidos em 'rotulo': {invalid}")

    # ------------------------------------------------------------------
    # Parsing de video_frame
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_video_frame(video_frame: str) -> tuple[str, int]:
        """Extrai (video_id, frame_id) de strings como 'v12_f76'."""
        parts = video_frame.split("_")
        video_id = parts[0][1:]   # remove o 'v'
        frame_id = int(parts[1][1:])  # remove o 'f'
        return video_id, frame_id

    def _get_video_path(self, video_id: str) -> str:
        filename = self.video_filename_pattern.format(video_id=video_id)
        # Path normaliza separadores automaticamente no Windows
        return str(Path(self.videos_root) / filename)

    def _get_metadata(self, video_frame: str) -> pd.Series:
        mask = self.metadata_df["video_frame"] == video_frame
        # Filtra por rotulador se a coluna existir, para evitar ambiguidade
        # quando o mesmo video_frame aparece para múltiplos rotuladores
        labeler_col = next(
            (c for c in ("labeler", "rotulador") if c in self.metadata_df.columns),
            None,
        )
        if labeler_col is not None:
            mask = mask & (self.metadata_df[labeler_col] == self.labeler)
        row = self.metadata_df[mask]
        if row.empty:
            raise KeyError(
                f"video_frame '{video_frame}' + labeler '{self.labeler}' "
                f"não encontrado em metadata_df"
            )
        return row.iloc[0]

    # ------------------------------------------------------------------
    # Carregamento de dados do frame atual
    # ------------------------------------------------------------------

    def _load_current_frame_data(self):
        """Retorna (image_rgb, points_or_None, mask_or_None, meta) para o frame atual."""
        row = self.review_df.iloc[self.current_index]
        video_frame = row["video_frame"]
        rotulo = row["rotulo"]

        video_id, frame_id = self._parse_video_frame(video_frame)
        video_path = self._get_video_path(video_id)
        meta = self._get_metadata(video_frame)

        # Frame do vídeo
        frame_bgr = get_frame_from_video(video_path, frame_id)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        points = None
        mask = None

        file_path = meta["file_path"]

        if rotulo == "point":
            if meta.get("has_points", False):
                try:
                    pts = load_points(file_path)   # função externa — import do chamador
                    # napari Points usa ordem (row, col) = (y, x)
                    points = pts[:, ::-1]  # converte [x,y] → [y,x]
                except Exception as e:
                    print(f"[AVISO] Não foi possível carregar pontos: {e}")

        elif rotulo == "mask":
            if meta.get("has_mask", False):
                mask_path = str(Path(file_path) / "Mask.tif")
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        mask = tifffile.imread(mask_path)
                except Exception as e:
                    print(f"[AVISO] Não foi possível carregar máscara: {e}")

        return frame_rgb, points, mask, meta

    # ------------------------------------------------------------------
    # Atualização do viewer napari
    # ------------------------------------------------------------------

    def _update_viewer(self):
        row = self.review_df.iloc[self.current_index]
        rotulo = row["rotulo"]
        video_frame = row["video_frame"]
        revisado = bool(row.get("revisado", False))

        frame_rgb, points, mask, meta = self._load_current_frame_data()
        h, w = frame_rgb.shape[:2]

        # Limpa layers anteriores
        self.viewer.layers.clear()

        # Layer de imagem base
        self._img_layer = self.viewer.add_image(
            frame_rgb,
            name="Frame",
            rgb=True,
        )

        if rotulo == "point":
            # Inicializa com pontos existentes ou dois pontos centrais
            if points is not None and len(points) > 0:
                initial_points = points  # já em (y, x)
            else:
                # Dois pontos padrão no centro superior e inferior
                initial_points = np.array([
                    [h * 0.33, w * 0.5],
                    [h * 0.66, w * 0.5],
                ])

            self._points_layer = self.viewer.add_points(
                initial_points,
                name="Points",
                size=4,
                face_color=["#FF4444", "#4444FF"],
                symbol="disc",
            )
            self._points_layer.mode = "select"
            self._mask_layer = None

        elif rotulo == "mask":
            # Inicializa com máscara existente ou vazia
            if mask is not None:
                initial_mask = (mask > 0).astype(np.uint8)
            else:
                initial_mask = np.zeros((h, w), dtype=np.uint8)

            self._mask_layer = self.viewer.add_labels(
                initial_mask,
                name="Mask",
                opacity=0.5,
            )
            # Define cor lime para label=1 (API varia por versão do napari)
            try:
                # napari >= 0.4.18
                self._mask_layer.colormap = {
                    None: "transparent",
                    1:    "limegreen",
                }
            except Exception:
                try:
                    # napari < 0.4.18
                    self._mask_layer.color = {1: "limegreen"}
                except Exception:
                    pass  # cor padrão do napari se nenhuma API funcionar
            self._mask_layer.mode = "paint"
            self._mask_layer.brush_size = self.brush_slider_value
            self._points_layer = None

        # Atualiza painel lateral
        has_existing = (
            bool(meta.get("has_points", False)) if rotulo == "point"
            else bool(meta.get("has_mask", False))
        )
        self.controls.update_info(
            index=self.current_index,
            total=len(self.review_df),
            video_frame=video_frame,
            rotulo=rotulo,
            motivo=row["motivo"],
            revisado=revisado,
            has_existing=has_existing,
        )

        self.viewer.title = f"Label Editor — {video_frame} [{rotulo}]"
        self.viewer.reset_view()

    # ------------------------------------------------------------------
    # Ações dos botões
    # ------------------------------------------------------------------

    def save_and_next(self):
        """Valida, salva o rótulo editado, atualiza o CSV e avança."""
        row = self.review_df.iloc[self.current_index]
        rotulo = row["rotulo"]
        video_frame = row["video_frame"]
        video_id, frame_id = self._parse_video_frame(video_frame)
        meta = self._get_metadata(video_frame)
        file_path = meta["file_path"]

        # Se output_root é None: sobrescreve na pasta original (file_path)
        # Caso contrário: salva em output_root / video_frame
        if self.output_root is None:
            output_dir = str(Path(file_path))
        else:
            output_dir = str(Path(self.output_root) / video_frame)

        if rotulo == "point":
            pts_yx = self._points_layer.data  # (N, 2) em [y, x]
            if len(pts_yx) != 2:
                QMessageBox.warning(
                    self.controls,
                    "Validação",
                    f"São necessários exatamente 2 pontos (C2 e C4). "
                    f"Atualmente há {len(pts_yx)} ponto(s).",
                )
                return
            pts_xy = pts_yx[:, ::-1]  # converte [y,x] → [x,y]

            # Caminho do CSV original para preservar colunas secundárias
            original_csv = str(Path(meta["file_path"]) / "Results.csv")
            # Se não há CSV original (criando do zero), usa None — save_points lida com isso
            if not os.path.exists(original_csv):
                original_csv = None

            save_points(pts_xy, output_dir, original_csv_path=original_csv)
            print(f"[✔] Pontos salvos em {output_dir}/Results.csv")

        elif rotulo == "mask":
            # np.asarray garante cópia real do array (evita máscara vazia
            # causada por lazy evaluation de algumas versões do napari)
            raw = np.asarray(self._mask_layer.data)
            mask_data = (raw > 0).astype(np.uint8)
            n_pixels = int(mask_data.sum())
            print(f"    Pixels marcados na máscara: {n_pixels}")
            if n_pixels == 0:
                from qtpy.QtWidgets import QMessageBox
                resp = QMessageBox.question(
                    self.controls,
                    "Máscara vazia",
                    "A máscara está vazia. Deseja salvar mesmo assim?",
                )
                from qtpy.QtWidgets import QMessageBox as _QMB
                if resp != _QMB.Yes:
                    return
            # Garante resolução igual à do frame do vídeo
            frame_shape = self._img_layer.data.shape
            save_mask(mask_data, output_dir, target_shape=frame_shape)
            print(f"[✔] Máscara salva em {output_dir}/Mask.tif")

        # Atualiza o dataframe
        ts = datetime.now().isoformat(timespec="seconds")
        self.review_df.at[self.current_index, "revisado"] = True
        self.review_df.at[self.current_index, "revisado_em"] = ts

        # Persiste o CSV atualizado
        self.review_df.to_csv(self.review_csv_path, index=False)
        print(f"[✔] CSV atualizado: {self.review_csv_path}")

        self._advance()

    def skip_frame(self):
        """Pula o frame atual sem salvar."""
        print(f"[→] Frame {self.review_df.iloc[self.current_index]['video_frame']} pulado.")
        self._advance()

    def go_next(self):
        if self.current_index < len(self.review_df) - 1:
            self.current_index += 1
            self._update_viewer()

    def go_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._update_viewer()

    def _advance(self):
        if self.current_index < len(self.review_df) - 1:
            self.current_index += 1
            self._update_viewer()
        else:
            QMessageBox.information(
                self.controls,
                "Concluído",
                "✅ Todos os frames do dataframe foram processados!\n"
                f"CSV salvo em: {self.review_csv_path}",
            )
            print("[✔] Todos os frames processados.")

    def set_brush_size(self, size: int):
        self.brush_slider_value = size
        if self._mask_layer is not None:
            self._mask_layer.brush_size = size

    # ------------------------------------------------------------------
    # Ponto de entrada
    # ------------------------------------------------------------------

    def run(self):
        """Inicializa o viewer napari e entra no loop de eventos.

        Quando chamado de dentro de um Jupyter/VSCode notebook, ativa
        automaticamente o backend Qt no IPython (%gui qt) para evitar
        o conflito de event loops (RuntimeError: Cannot activate multiple
        GUI eventloops).
        """
        if len(self.review_df) == 0:
            print("[AVISO] O dataframe de revisão está vazio. Nada a fazer.")
            return

        self.brush_slider_value = 10

        # --- Compatibilidade com Jupyter / VSCode notebook ---
        # O IPython já possui um event loop; napari precisa compartilhá-lo
        # via %gui qt em vez de criar o seu próprio com napari.run().
        _ipy = _get_ipython() if _get_ipython is not None else None
        _in_notebook = _ipy is not None and hasattr(_ipy, "enable_gui")
        if _in_notebook:
            # Só ativa o backend Qt se ainda não estiver ativo —
            # chamar enable_gui quando já está ativo levanta RuntimeError
            _active = getattr(_ipy, "active_eventloop", None)
            if _active != "qt":
                _ipy.enable_gui("qt")

        self.viewer = napari.Viewer(title="Label Editor")

        # Painel de controle lateral
        self.controls = LabelEditorControls(self)
        self.viewer.window.add_dock_widget(
            self.controls,
            name="Controles",
            area="right",
        )

        # Referência ao slider do painel
        self.brush_slider_value = self.controls.brush_slider.value()

        # Carrega o primeiro frame
        self._update_viewer()

        # Em notebook: o event loop já está rodando no IPython, não chama napari.run()
        # Em script puro: napari.run() bloqueia até fechar a janela e salva o CSV.
        if not _in_notebook:
            napari.run()
            self.review_df.to_csv(self.review_csv_path, index=False)
            print(f"[✔] Sessão encerrada. CSV final salvo em: {self.review_csv_path}")
        else:
            print("[ℹ] Janela napari aberta. Salve o CSV ao terminar chamando:")
            print(f"    editor.review_df.to_csv(r'{self.review_csv_path}', index=False)")
            print("    (ou use os botões — o CSV é salvo a cada 'Salvar e Avançar')")


# ---------------------------------------------------------------------------
# Função pública de entrada
# ---------------------------------------------------------------------------

def run_label_editor(
    review_csv_path: str,
    video_frame_metadata_df: pd.DataFrame,
    videos_root: str,
    labeler: str,
    output_root: str | None = None,
    video_filename_pattern: str = "video_{video_id}.avi",
) -> pd.DataFrame:
    """
    Abre a interface interativa de edição de rótulos.

    Parâmetros
    ----------
    review_csv_path : str
        Caminho para o CSV do rotulador.
        Colunas obrigatórias: ``video_frame``, ``rotulo``, ``motivo``.
    video_frame_metadata_df : pd.DataFrame
        Metadados de cada frame.
        Colunas obrigatórias: ``video_frame``, ``file_path``,
        ``has_mask``, ``has_points``.
    videos_root : str
        Pasta raiz com os arquivos .avi.
    labeler : str
        Identificador do rotulador (ex: "VR", "AM"). Usado para filtrar
        a linha correta em video_frame_metadata_df quando o mesmo
        video_frame aparece para múltiplos rotuladores.
    output_root : str or None, opcional
        Pasta de saída onde os rótulos editados serão salvos.
        Estrutura: ``output_root/{video_frame}/Results.csv`` ou ``Mask.tif``.
        Se None, sobrescreve os arquivos originais em ``file_path`` de cada frame.
    video_filename_pattern : str, opcional
        Padrão do nome do arquivo de vídeo. Use ``{video_id}`` como
        placeholder. Padrão: ``"video_{video_id}.avi"``.

    Retorna
    -------
    pd.DataFrame
        O dataframe de revisão atualizado (com colunas ``revisado`` e
        ``revisado_em`` preenchidas para os frames salvos).

    Exemplo
    -------
    >>> import pandas as pd
    >>> from label_editor import run_label_editor
    >>>
    >>> metadata = pd.read_csv("metadata.csv")
    >>> df_final = run_label_editor(
    ...     review_csv_path         = "revisao_rotulador1.csv",
    ...     video_frame_metadata_df = metadata,
    ...     videos_root             = "/data/videos",
    ...     output_root             = "/data/revisados",
    ...     video_filename_pattern  = "video_{video_id}.avi",
    ... )
    >>> print(df_final[["video_frame", "rotulo", "revisado", "revisado_em"]])
    """
    app = LabelEditorApp(
        review_csv_path=review_csv_path,
        video_frame_metadata_df=video_frame_metadata_df,
        videos_root=videos_root,
        output_root=output_root,
        labeler=labeler,
        video_filename_pattern=video_filename_pattern,
    )
    app.run()
    # Expõe o objeto app no módulo para acesso ao review_df em notebooks
    # (onde napari.run() não bloqueia e o df é atualizado de forma assíncrona)
    import sys as _sys
    _sys.modules[__name__].__dict__["_last_editor"] = app
    return app.review_df
