import sys
import os
import json
import math
import time
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QDoubleSpinBox, QSpinBox, QMessageBox, QComboBox, 
                             QTextEdit, QLineEdit, QInputDialog, QFileDialog, 
                             QGroupBox, QCheckBox, QGridLayout, QDialog, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

def compute_rand_indices(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    le = LabelEncoder()
    true_l = le.fit_transform(labels_true)
    pred_l = LabelEncoder().fit_transform(labels_pred)
    true_pairs = (true_l[:, None] == true_l[None, :])
    pred_pairs = (pred_l[:, None] == pred_l[None, :])
    iu = np.triu_indices(len(true_l), k=1)
    tp = np.sum(true_pairs[iu] & pred_pairs[iu])
    tn = np.sum((~true_pairs[iu]) & (~pred_pairs[iu]))
    total = len(iu[0])
    ri = (tp + tn) / total if total > 0 else 0
    return ari, ri

def generate_weight_combinations(n_vars, target_sum, step):
    target_int = int(round(target_sum / step))
    def _gen(n, current_target):
        if n == 1:
            yield [round(current_target * step, 4)]
        else:
            for i in range(current_target + 1):
                for tail in _gen(n - 1, current_target - i):
                    yield [round(i * step, 4)] + tail
    for combo in _gen(n_vars, target_int):
        yield combo

class OptimizerThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list, list, list, list) 

    def __init__(self, matrices, fixed_weights, step, n_clusters, algo_name, true_labels, max_time_sec):
        super().__init__()
        self.matrices = matrices
        self.fixed_weights = fixed_weights
        self.step = step
        self.n_clusters = n_clusters
        self.algo_name = algo_name
        self.true_labels = true_labels
        self.max_time_sec = max_time_sec
        self._is_running = True

    def run(self):
        unlocked_indices = [i for i, w in enumerate(self.fixed_weights) if w is None]
        fixed_sum = sum(w for w in self.fixed_weights if w is not None)
        target_sum = 1.0 - fixed_sum

        if target_sum < -1e-5:
            self.finished.emit([], [], [], [])
            return
            
        if not unlocked_indices:
            self.finished.emit(self.fixed_weights, [], [], [])
            return

        combinations = list(generate_weight_combinations(len(unlocked_indices), target_sum, self.step))
        total_iters = len(combinations)
        
        best_ari = -1.0
        best_weights = self.fixed_weights.copy()
        
        history_ari = []
        history_ri = []
        history_iters = []

        start_time = time.time()

        for iter_idx, combo in enumerate(combinations):
            if not self._is_running or (time.time() - start_time) > self.max_time_sec:
                break
                
            current_weights = self.fixed_weights.copy()
            for i, idx in enumerate(unlocked_indices):
                current_weights[idx] = combo[i]
                
            D_total = sum(w * mat for w, mat in zip(current_weights, self.matrices) if w > 0)
            D_total = np.clip(D_total, 0, None)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.algo_name == "Agglomerative (average)":
                    clustering = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage='average')
                    preds = clustering.fit_predict(D_total)
                elif self.algo_name == "Agglomerative (complete)":
                    clustering = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage='complete')
                    preds = clustering.fit_predict(D_total)
                elif self.algo_name == "Agglomerative (single)":
                    clustering = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage='single')
                    preds = clustering.fit_predict(D_total)
                elif self.algo_name == "K-Medoids" and HAS_KMEDOIDS:
                    clustering = KMedoids(n_clusters=self.n_clusters, metric='precomputed')
                    preds = clustering.fit_predict(D_total)
                elif self.algo_name == "Spectral Clustering":
                    S = 1.0 - D_total
                    np.fill_diagonal(S, 1.0)
                    clustering = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed')
                    preds = clustering.fit_predict(S)
                else:
                    preds = np.zeros(len(D_total))

            ari, ri = compute_rand_indices(self.true_labels, preds)
            
            history_ari.append(ari)
            history_ri.append(ri)
            history_iters.append(iter_idx + 1)
            
            if ari > best_ari:
                best_ari = ari
                best_weights = current_weights.copy()
                
            if iter_idx % max(1, total_iters // 100) == 0 or iter_idx == total_iters - 1:
                self.progress.emit(iter_idx + 1, total_iters)

        self.finished.emit(best_weights, history_ari, history_ri, history_iters)

    def stop(self):
        self._is_running = False

class AdvancedWeightsWidget(QGroupBox):
    def __init__(self, title, features_config):
        super().__init__(title)
        self.features_config = features_config
        self.rows = []
        self.is_updating = False
        
        layout = QGridLayout(self)
        layout.addWidget(QLabel("<b>Критерій</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Вага</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Фікс.</b>"), 0, 2)
        layout.addWidget(QLabel("<b>UMAP</b>"), 0, 3)
        layout.addWidget(QLabel("<b>Розмір</b>"), 0, 4)
        
        for i, config in enumerate(features_config):
            row_idx = i + 1
            lbl = QLabel(config['label'])
            layout.addWidget(lbl, row_idx, 0)
            
            sb_weight = QDoubleSpinBox()
            sb_weight.setRange(0.0, 1.0)
            sb_weight.setSingleStep(0.05)
            sb_weight.setValue(config['default_w'])
            sb_weight.valueChanged.connect(lambda val, idx=i: self.on_weight_change(idx))
            layout.addWidget(sb_weight, row_idx, 1)
            
            cb_lock = QCheckBox()
            layout.addWidget(cb_lock, row_idx, 2)
            
            cb_umap = QCheckBox()
            cb_umap.setEnabled(config['compressible'])
            layout.addWidget(cb_umap, row_idx, 3)
            
            sb_dim = QSpinBox()
            sb_dim.setRange(2, 500)
            sb_dim.setValue(config.get('default_dim', 10))
            sb_dim.setEnabled(config['compressible'])
            
            cb_umap.toggled.connect(sb_dim.setEnabled)
            cb_umap.setChecked(config.get('default_umap', False))
            
            layout.addWidget(sb_dim, row_idx, 4)
            
            self.rows.append({
                'label': config['label'],
                'sb_weight': sb_weight,
                'cb_lock': cb_lock,
                'cb_umap': cb_umap,
                'sb_dim': sb_dim,
                'metric': config['metric']
            })
            
        self.prev_weights = self.get_weights()

    def get_weights(self):
        return [row['sb_weight'].value() for row in self.rows]
        
    def set_weights(self, weights):
        self.is_updating = True
        for row, w in zip(self.rows, weights):
            row['sb_weight'].setValue(w)
        self.prev_weights = self.get_weights()
        self.is_updating = False

    def get_settings(self):
        settings = []
        for row in self.rows:
            settings.append({
                'label': row['label'],
                'weight': row['sb_weight'].value(),
                'locked': row['cb_lock'].isChecked(),
                'use_umap': row['cb_umap'].isChecked(),
                'umap_dim': row['sb_dim'].value(),
                'metric': row['metric']
            })
        return settings

    def on_weight_change(self, changed_idx):
        if self.is_updating:
            return
        self.is_updating = True
        new_val = self.rows[changed_idx]['sb_weight'].value()
        locked = [row['cb_lock'].isChecked() for row in self.rows]
        effectively_locked = locked.copy()
        effectively_locked[changed_idx] = True
        locked_sum = sum(self.prev_weights[j] for j in range(len(self.rows)) if locked[j] and j != changed_idx)
        
        if new_val + locked_sum > 1.0:
            new_val = 1.0 - locked_sum
            self.rows[changed_idx]['sb_weight'].setValue(new_val)
            
        remaining = 1.0 - (new_val + locked_sum)
        unlocked_indices = [j for j in range(len(self.rows)) if not effectively_locked[j]]
        
        if unlocked_indices:
            unlocked_sum = sum(self.prev_weights[j] for j in unlocked_indices)
            for j in unlocked_indices:
                if unlocked_sum > 0:
                    self.rows[j]['sb_weight'].setValue(remaining * (self.prev_weights[j] / unlocked_sum))
                else:
                    self.rows[j]['sb_weight'].setValue(remaining / len(unlocked_indices))
        else:
            self.rows[changed_idx]['sb_weight'].setValue(1.0 - locked_sum)
            
        self.prev_weights = self.get_weights()
        self.is_updating = False

class ClusteringPipelineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Пайплайн Кластеризації (Advanced UMAP + Grid Search + Colorbar Fix)")
        self.resize(1600, 900)

        self.vectorized_dir = "vectorized"
        self.original_dir = "original"
        self.saves_dir = "saved_clusterings"
        
        if not os.path.exists(self.saves_dir):
            os.makedirs(self.saves_dir)
        
        self.users_data = self.load_data(os.path.join(self.vectorized_dir, "users_vectorized.json"))
        self.courses_data = self.load_data(os.path.join(self.vectorized_dir, "courses_vectorized.json"))
        self.users_orig_data = self.load_data(os.path.join(self.original_dir, "users_raw.json"))
        self.courses_orig_data = self.load_data(os.path.join(self.original_dir, "courses_raw.json"))

        self.current_courses_clusters = []
        self.current_users_clusters = []
        self.courses_embedding = None
        self.users_embedding = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tab_courses = QWidget()
        self.tab_users = QWidget()
        self.tab_data_view = QWidget()

        self.tabs.addTab(self.tab_courses, "Курси")
        self.tabs.addTab(self.tab_users, "Користувачі")
        self.tabs.addTab(self.tab_data_view, "Менеджер даних")

        self.status_bar = QLabel("Готово")
        self.statusBar().addWidget(self.status_bar)

        self.setup_courses_tab()
        self.setup_users_tab()
        self.setup_data_view_tab()
        
        self.calculate_base_embeddings()
        
        if self.courses_embedding is not None: self.draw_courses_plot()
        if self.users_embedding is not None: self.draw_users_plot()

    def set_loading_state(self, message):
        self.status_bar.setText(message)
        QApplication.processEvents()

    def load_data(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def extract_and_pad_features(self, df, text_col, tag_cols, num_cols):
        if text_col in df.columns:
            raw_text = df[text_col].tolist()
            max_len = max((len(v) if isinstance(v, list) else 0) for v in raw_text)
            padded_text = []
            for v in raw_text:
                if isinstance(v, list):
                    padded_text.append(v + [0.0] * (max_len - len(v)))
                else:
                    padded_text.append([0.0] * max_len)
            text_features = np.array(padded_text)
        else:
            text_features = np.empty((len(df), 0))

        available_tags = [c for c in tag_cols if c in df.columns]
        tags_features = df[available_tags].fillna(0).values
        available_nums = [c for c in num_cols if c in df.columns]
        num_features = df[available_nums].fillna(0).values

        return np.hstack([text_features, tags_features, num_features])

    def setup_courses_tab(self):
        layout = QHBoxLayout(self.tab_courses)
        control_frame = QWidget()
        control_frame.setFixedWidth(400)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(0, 0, 15, 0)
        layout.addWidget(control_frame)

        c_features = [
            {'label': 'Текст (TF-IDF)', 'default_w': 0.4, 'compressible': True, 'default_umap': True, 'default_dim': 50, 'metric': 'cosine'},
            {'label': 'Теги (One-Hot)', 'default_w': 0.3, 'compressible': True, 'default_umap': False, 'default_dim': 10, 'metric': 'jaccard'},
            {'label': 'Формати', 'default_w': 0.2, 'compressible': True, 'default_umap': False, 'default_dim': 5, 'metric': 'cosine'},
            {'label': 'Числові', 'default_w': 0.1, 'compressible': False, 'default_umap': False, 'metric': 'euclidean'}
        ]
        self.c_weights_widget = AdvancedWeightsWidget("Критерії кластеризації", c_features)
        control_layout.addWidget(self.c_weights_widget)

        algo_group = QGroupBox("Алгоритм")
        vbox_a = QVBoxLayout()
        vbox_a.addWidget(QLabel("Метод:"))
        self.c_algo = QComboBox()
        algos = ["Agglomerative (average)", "Agglomerative (complete)", "Agglomerative (single)"]
        if HAS_KMEDOIDS: algos.append("K-Medoids")
        self.c_algo.addItems(algos)
        self.c_algo.setCurrentText("Agglomerative (average)")
        vbox_a.addWidget(self.c_algo)
        
        vbox_a.addWidget(QLabel("Кількість кластерів:"))
        self.c_n_clusters = QSpinBox()
        self.c_n_clusters.setRange(2, 50)
        self.c_n_clusters.setValue(10)
        vbox_a.addWidget(self.c_n_clusters)
        algo_group.setLayout(vbox_a)
        control_layout.addWidget(algo_group)

        view_group = QGroupBox("Відображення та Метрики")
        vbox_v = QVBoxLayout()
        vbox_v.addWidget(QLabel("Колір точок:"))
        self.c_color_mode = QComboBox()
        self.c_color_mode.addItems(["Кластер", "Теги", "Формати матеріалів курсу", "Складність"])
        self.c_color_mode.currentIndexChanged.connect(self.draw_courses_plot)
        vbox_v.addWidget(self.c_color_mode)
        
        vbox_v.addWidget(QLabel("Радіус Hover:"))
        self.c_hover_radius = QDoubleSpinBox()
        self.c_hover_radius.setRange(0.1, 10.0)
        self.c_hover_radius.setSingleStep(0.1)
        self.c_hover_radius.setValue(0.5)
        vbox_v.addWidget(self.c_hover_radius)
        
        self.c_ari_label = QLabel("Adjusted Rand Index: N/A")
        self.c_ri_label = QLabel("Rand Index: N/A")
        self.c_ari_label.setStyleSheet("font-weight: bold; color: #003366;")
        self.c_ri_label.setStyleSheet("font-weight: bold; color: #333333;")
        vbox_v.addWidget(self.c_ari_label)
        vbox_v.addWidget(self.c_ri_label)
        view_group.setLayout(vbox_v)
        control_layout.addWidget(view_group)

        btn_update = QPushButton("Розрахувати кластери")
        btn_update.setMinimumHeight(35)
        btn_update.clicked.connect(self.process_courses)
        control_layout.addWidget(btn_update)
        
        tune_layout = QHBoxLayout()
        btn_autotune = QPushButton("🤖 Авто-підбір ваг")
        btn_autotune.setStyleSheet("background-color: #2b5797; color: white; font-weight: bold;")
        btn_autotune.setMinimumHeight(35)
        btn_autotune.clicked.connect(self.autotune_courses)
        
        self.c_tune_time = QSpinBox()
        self.c_tune_time.setRange(1, 120)
        self.c_tune_time.setValue(10)
        self.c_tune_time.setSuffix(" хв")
        
        tune_layout.addWidget(btn_autotune)
        tune_layout.addWidget(self.c_tune_time)
        control_layout.addLayout(tune_layout)
        
        btn_save = QPushButton("Зберегти кластеризацію ЯК...")
        btn_save.clicked.connect(self.save_courses)
        control_layout.addWidget(btn_save)
        control_layout.addStretch()
        
        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot_frame, 1)
        self.fig_courses = Figure(figsize=(9, 6))
        self.canvas_courses = FigureCanvas(self.fig_courses)
        plot_layout.addWidget(self.canvas_courses)
        self.ax_courses = self.fig_courses.add_subplot(111)
        
        self.annot_courses = self.ax_courses.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="white", alpha=0.9), zorder=1000, clip_on=False)
        self.annot_courses.set_visible(False)
        self.hover_circle_courses = Circle((0,0), 0.1, fill=False, color='red', linestyle='--', zorder=999)
        self.hover_circle_courses.set_visible(False)
        self.ax_courses.add_patch(self.hover_circle_courses)
        self.canvas_courses.mpl_connect("motion_notify_event", self.hover_courses)

    def extract_course_matrices(self, settings):
        df = pd.DataFrame(self.courses_data)
        matrices = []
        
        s = settings[0]
        if 'TF-IDF_vector' in df.columns:
            raw = df['TF-IDF_vector'].tolist()
            maxlen = max((len(v) if isinstance(v, list) else 0) for v in raw)
            padded = np.array([(v + [0.0]*(maxlen-len(v)) if isinstance(v,list) else [0.0]*maxlen) for v in raw])
            if padded.shape[1] > 0:
                if s['use_umap']:
                    red = umap.UMAP(n_components=s['umap_dim'], metric=s['metric'])
                    emb = red.fit_transform(padded)
                    d_mat = pairwise_distances(emb, metric='euclidean')
                else:
                    d_mat = pairwise_distances(padded, metric=s['metric'])
            else: d_mat = np.zeros((len(df), len(df)))
        else: d_mat = np.zeros((len(df), len(df)))
        matrices.append(self.sanitize_matrix(d_mat))
        
        s = settings[1]
        tag_cols = [c for c in df.columns if c.startswith('Тег_')]
        if len(tag_cols) > 0:
            t_mat = df[tag_cols].fillna(0).astype(bool).values
            if s['use_umap']:
                red = umap.UMAP(n_components=s['umap_dim'], metric=s['metric'])
                emb = red.fit_transform(t_mat)
                d_mat = pairwise_distances(emb, metric='euclidean')
            else:
                d_mat = pairwise_distances(t_mat, metric=s['metric'])
        else: d_mat = np.zeros((len(df), len(df)))
        matrices.append(self.sanitize_matrix(d_mat))
        
        s = settings[2]
        fmt_cols = [c for c in df.columns if c.startswith('ФорматКурсу_')]
        if len(fmt_cols) > 0:
            f_mat = df[fmt_cols].fillna(0).values
            if s['use_umap']:
                red = umap.UMAP(n_components=s['umap_dim'], metric=s['metric'])
                emb = red.fit_transform(f_mat)
                d_mat = pairwise_distances(emb, metric='euclidean')
            else:
                d_mat = pairwise_distances(f_mat, metric=s['metric'])
        else: d_mat = np.zeros((len(df), len(df)))
        matrices.append(self.sanitize_matrix(d_mat))
        
        s = settings[3]
        num_cols = ['Ціна_norm', 'Складність_norm', 'Тривалість_norm']
        if len(num_cols) > 0:
            n_mat = df[num_cols].fillna(0).values
            d_num_raw = pairwise_distances(n_mat, metric='euclidean')
            max_d = d_num_raw.max()
            d_mat = d_num_raw / max_d if max_d > 0 else d_num_raw
        else: d_mat = np.zeros((len(df), len(df)))
        matrices.append(self.sanitize_matrix(d_mat))
        
        return matrices

    def process_courses(self):
        settings = self.c_weights_widget.get_settings()
        weights = [s['weight'] for s in settings]
        if abs(sum(weights) - 1.0) > 0.001:
            QMessageBox.warning(self, "Увага", "Сума коефіцієнтів має дорівнювати 1.0!")
            return
        if not self.courses_data: return

        self.set_loading_state("Розрахунок матриць відстаней...")
        try:
            matrices = self.extract_course_matrices(settings)
            D_total = sum(w * m for w, m in zip(weights, matrices))
            
            n_clust = self.c_n_clusters.value()
            algo = self.c_algo.currentText()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if algo == "Agglomerative (average)": clustering = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed', linkage='average')
                elif algo == "Agglomerative (complete)": clustering = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed', linkage='complete')
                elif algo == "Agglomerative (single)": clustering = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed', linkage='single')
                elif algo == "K-Medoids" and HAS_KMEDOIDS: clustering = KMedoids(n_clusters=n_clust, metric='precomputed')
                else: raise Exception("Невідомий алгоритм")
                self.current_courses_clusters = clustering.fit_predict(D_total)
            
            true_tags = []
            for item in self.courses_orig_data:
                tags = item.get('Теги', [])
                true_tags.append(tags[0] if tags else "None")
                
            ari, ri = compute_rand_indices(true_tags, self.current_courses_clusters)
            self.c_ari_label.setText(f"Adjusted Rand Index: {ari:.4f}")
            self.c_ri_label.setText(f"Rand Index: {ri:.4f}")

            self.c_color_mode.setCurrentIndex(0) 
            self.draw_courses_plot()
        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))
        finally:
            self.set_loading_state("Готово")

    def autotune_courses(self):
        settings = self.c_weights_widget.get_settings()
        weights = [s['weight'] for s in settings]
        locked = [s['weight'] if s['locked'] else None for s in settings]
        self.set_loading_state("Попередня генерація матриць для оптимізації...")
        matrices = self.extract_course_matrices(settings)
        true_tags = []
        for item in self.courses_orig_data:
            tags = item.get('Теги', [])
            true_tags.append(tags[0] if tags else "None")
        time_limit_sec = self.c_tune_time.value() * 60
        self.start_optimizer(matrices, locked, self.c_n_clusters.value(), self.c_algo.currentText(), true_tags, True, time_limit_sec)

    def draw_courses_plot(self):
        if self.courses_embedding is None: return
        self.fig_courses.clear()
        self.ax_courses = self.fig_courses.add_subplot(111)
        mode = self.c_color_mode.currentText()
        self.ax_courses.set_title(f"UMAP: Курси (Колір: {mode})")
        
        if mode == "Кластер":
            uniques, counts = np.unique(self.current_courses_clusters, return_counts=True)
            n_clusters = len(uniques)
            if n_clusters > 0:
                mapped_clusters = np.array([np.where(uniques == c)[0][0] for c in self.current_courses_clusters])
                cmap = plt.get_cmap('tab20', n_clusters)
                scatter = self.ax_courses.scatter(
                    self.courses_embedding[:, 0], self.courses_embedding[:, 1], 
                    c=mapped_clusters, cmap=cmap, vmin=-0.5, vmax=n_clusters-0.5, s=50, alpha=0.8
                )
                self.cbar_courses = self.fig_courses.colorbar(scatter, ax=self.ax_courses, ticks=range(n_clusters))
                self.cbar_courses.set_ticklabels([f"Кл {int(c)} ({n})" for c, n in zip(uniques, counts)])
        else:
            is_list = mode in ["Теги", "Формати матеріалів курсу"]
            c_arr, c_map = self.get_colors_and_legend(self.courses_orig_data, mode, is_list)
            self.ax_courses.scatter(self.courses_embedding[:, 0], self.courses_embedding[:, 1], c=c_arr, s=50, alpha=0.8)
            leg = [Patch(facecolor=c_map[k], label=k) for k in c_map]
            self.ax_courses.legend(handles=leg, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
            
        self.fig_courses.subplots_adjust(right=0.72 if mode != 'Кластер' else 0.8, left=0.1, top=0.9, bottom=0.1)
        
        self.annot_courses = self.ax_courses.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="white", alpha=0.9), zorder=1000, clip_on=False)
        self.annot_courses.set_visible(False)
        self.hover_circle_courses = Circle((0,0), self.c_hover_radius.value(), fill=False, color='red', linestyle='--', zorder=999)
        self.hover_circle_courses.set_visible(False)
        self.ax_courses.add_patch(self.hover_circle_courses)
        self.canvas_courses.draw()

    def hover_courses(self, event):
        if event.inaxes == self.ax_courses and self.courses_embedding is not None:
            x, y = event.xdata, event.ydata
            radius = self.c_hover_radius.value()
            distances = np.sqrt((self.courses_embedding[:, 0] - x)**2 + (self.courses_embedding[:, 1] - y)**2)
            count = np.sum(distances <= radius)
            
            if count > 0:
                xlim = self.ax_courses.get_xlim()
                if x > (xlim[0] + xlim[1]) / 2:
                    self.annot_courses.set_position((-15, 15))
                    self.annot_courses.set_horizontalalignment('right')
                else:
                    self.annot_courses.set_position((15, 15))
                    self.annot_courses.set_horizontalalignment('left')

                self.annot_courses.xy = (x, y)
                self.annot_courses.set_text(f"Курсів у r={radius}: {count}")
                self.annot_courses.set_visible(True)
                
                self.hover_circle_courses.center = (x, y)
                self.hover_circle_courses.set_radius(radius)
                self.hover_circle_courses.set_visible(True)
                self.canvas_courses.draw_idle()
            else:
                if self.annot_courses.get_visible():
                    self.annot_courses.set_visible(False)
                    self.hover_circle_courses.set_visible(False)
                    self.canvas_courses.draw_idle()
        else:
            if hasattr(self, 'annot_courses') and self.annot_courses.get_visible():
                self.annot_courses.set_visible(False)
                self.hover_circle_courses.set_visible(False)
                self.canvas_courses.draw_idle()

    def setup_users_tab(self):
        layout = QHBoxLayout(self.tab_users)
        control_frame = QWidget()
        control_frame.setFixedWidth(420)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(0, 0, 15, 0)
        layout.addWidget(control_frame)

        u_features = [
            {'label': 'Пройдені (Cos)', 'default_w': 0.3, 'compressible': True, 'default_umap': True, 'default_dim': 10, 'metric': 'cosine'},
            {'label': 'Бажані (Cos)', 'default_w': 0.3, 'compressible': True, 'default_umap': True, 'default_dim': 10, 'metric': 'cosine'},
            {'label': 'Формати (Jaccard)', 'default_w': 0.1, 'compressible': True, 'default_umap': False, 'default_dim': 5, 'metric': 'jaccard'},
            {'label': 'Числові (Euclid)', 'default_w': 0.1, 'compressible': False, 'metric': 'euclidean'},
            {'label': 'Освіта (Euclid)', 'default_w': 0.1, 'compressible': False, 'metric': 'euclidean'},
            {'label': 'Мета (Таблиця)', 'default_w': 0.1, 'compressible': False, 'metric': 'custom'}
        ]
        self.u_weights_widget = AdvancedWeightsWidget("Критерії кластеризації", u_features)
        control_layout.addWidget(self.u_weights_widget)

        algo_group = QGroupBox("Алгоритм")
        vbox_a = QVBoxLayout()
        vbox_a.addWidget(QLabel("Метод:"))
        self.u_algo = QComboBox()
        algos_u = ["Spectral Clustering", "Agglomerative (average)", "Agglomerative (complete)", "Agglomerative (single)"]
        if HAS_KMEDOIDS: algos_u.append("K-Medoids")
        self.u_algo.addItems(algos_u)
        self.u_algo.setCurrentText("Agglomerative (average)")
        vbox_a.addWidget(self.u_algo)
        
        vbox_a.addWidget(QLabel("Кількість кластерів:"))
        self.u_n_clusters = QSpinBox()
        self.u_n_clusters.setRange(2, 50)
        self.u_n_clusters.setValue(10)
        vbox_a.addWidget(self.u_n_clusters)
        algo_group.setLayout(vbox_a)
        control_layout.addWidget(algo_group)

        view_group = QGroupBox("Відображення та Метрики")
        vbox_v = QVBoxLayout()
        vbox_v.addWidget(QLabel("Колір точок:"))
        self.u_color_mode = QComboBox()
        self.u_color_mode.addItems(["Кластер", "User_Type", "Улюблені теми", "Мета навчання"])
        self.u_color_mode.currentIndexChanged.connect(self.draw_users_plot)
        vbox_v.addWidget(self.u_color_mode)
        
        vbox_v.addWidget(QLabel("Радіус Hover:"))
        self.u_hover_radius = QDoubleSpinBox()
        self.u_hover_radius.setRange(0.1, 10.0)
        self.u_hover_radius.setSingleStep(0.1)
        self.u_hover_radius.setValue(0.5)
        vbox_v.addWidget(self.u_hover_radius)
        
        self.u_ari_label = QLabel("Adjusted Rand Index: N/A")
        self.u_ri_label = QLabel("Rand Index: N/A")
        self.u_ari_label.setStyleSheet("font-weight: bold; color: #003366;")
        self.u_ri_label.setStyleSheet("font-weight: bold; color: #333333;")
        vbox_v.addWidget(self.u_ari_label)
        vbox_v.addWidget(self.u_ri_label)
        view_group.setLayout(vbox_v)
        control_layout.addWidget(view_group)

        btn_update = QPushButton("Розрахувати кластери")
        btn_update.setMinimumHeight(35)
        btn_update.clicked.connect(self.process_users)
        control_layout.addWidget(btn_update)
        
        tune_layout = QHBoxLayout()
        btn_autotune = QPushButton("🤖 Авто-підбір ваг")
        btn_autotune.setStyleSheet("background-color: #2b5797; color: white; font-weight: bold;")
        btn_autotune.setMinimumHeight(35)
        btn_autotune.clicked.connect(self.autotune_users)
        
        self.u_tune_time = QSpinBox()
        self.u_tune_time.setRange(1, 120)
        self.u_tune_time.setValue(10)
        self.u_tune_time.setSuffix(" хв")
        
        tune_layout.addWidget(btn_autotune)
        tune_layout.addWidget(self.u_tune_time)
        control_layout.addLayout(tune_layout)
        
        btn_save = QPushButton("Зберегти кластеризацію ЯК...")
        btn_save.clicked.connect(self.save_users)
        control_layout.addWidget(btn_save)
        control_layout.addStretch()

        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot_frame, 1)
        self.fig_users = Figure(figsize=(9, 6))
        self.canvas_users = FigureCanvas(self.fig_users)
        plot_layout.addWidget(self.canvas_users)
        self.ax_users = self.fig_users.add_subplot(111)
        
        self.annot_users = self.ax_users.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="white", alpha=0.9), zorder=1000, clip_on=False)
        self.annot_users.set_visible(False)
        self.hover_circle_users = Circle((0,0), 0.1, fill=False, color='red', linestyle='--', zorder=999)
        self.hover_circle_users.set_visible(False)
        self.ax_users.add_patch(self.hover_circle_users)
        self.canvas_users.mpl_connect("motion_notify_event", self.hover_users)

    def extract_user_matrices(self, settings):
        df = pd.DataFrame(self.users_data)
        N = len(df)
        matrices = []
        
        s = settings[0]
        p_cols = [c for c in df.columns if c.startswith('Пройдено_тег_')]
        if p_cols:
            mat = df[p_cols].fillna(0).values
            if s['use_umap']:
                red = umap.UMAP(n_components=s['umap_dim'], metric=s['metric'])
                emb = red.fit_transform(mat)
                d_mat = pairwise_distances(emb, metric='euclidean')
            else: d_mat = pairwise_distances(mat, metric=s['metric'])
        else: d_mat = np.zeros((N, N))
        matrices.append(self.sanitize_matrix(d_mat))
        
        s = settings[1]
        w_cols = [c for c in df.columns if c.startswith('Бажане_тег_')]
        if w_cols:
            mat = df[w_cols].fillna(0).values
            if s['use_umap']:
                red = umap.UMAP(n_components=s['umap_dim'], metric=s['metric'])
                emb = red.fit_transform(mat)
                d_mat = pairwise_distances(emb, metric='euclidean')
            else: d_mat = pairwise_distances(mat, metric=s['metric'])
        else: d_mat = np.zeros((N, N))
        matrices.append(self.sanitize_matrix(d_mat))
        
        s = settings[2]
        f_cols = [c for c in df.columns if c.startswith('Формат_')]
        if f_cols:
            mat = df[f_cols].fillna(0).astype(bool).values
            if s['use_umap']:
                red = umap.UMAP(n_components=s['umap_dim'], metric=s['metric'])
                emb = red.fit_transform(mat)
                d_mat = pairwise_distances(emb, metric='euclidean')
            else: d_mat = pairwise_distances(mat, metric=s['metric'])
        else: d_mat = np.zeros((N, N))
        matrices.append(self.sanitize_matrix(d_mat))
        
        n_cols = ['Час_навчання_norm', 'Витрати_norm', 'Вік_norm']
        if n_cols:
            mat = df[n_cols].fillna(0).values
            raw = pairwise_distances(mat, metric='euclidean')
            mx = raw.max()
            d_mat = raw/mx if mx>0 else raw
        else: d_mat = np.zeros((N,N))
        matrices.append(self.sanitize_matrix(d_mat))
        
        if 'Освіта_norm' in df.columns:
            mat = df[['Освіта_norm']].fillna(0).values
            raw = pairwise_distances(mat, metric='euclidean')
            mx = raw.max()
            d_mat = raw/mx if mx>0 else raw
        else: d_mat = np.zeros((N,N))
        matrices.append(self.sanitize_matrix(d_mat))
        
        goal_dist_table = np.array([[0.0, 1.0, 0.75], [1.0, 0.0, 0.25], [0.75, 0.25, 0.0]])
        goals = []
        for i, row in df.iterrows():
            if row.get('Мета навчання_Хобі', 0) == 1.0: goals.append(0)
            elif row.get('Мета навчання_Професійний розвиток', 0) == 1.0: goals.append(1)
            else: goals.append(2)
        goals = np.array(goals)
        d_goal = goal_dist_table[goals[:, None], goals[None, :]]
        matrices.append(self.sanitize_matrix(d_goal))
        
        return matrices

    def process_users(self):
        settings = self.u_weights_widget.get_settings()
        weights = [s['weight'] for s in settings]
        if abs(sum(weights) - 1.0) > 0.001:
            QMessageBox.warning(self, "Увага", "Сума коефіцієнтів має дорівнювати 1.0!")
            return
        if not self.users_data: return

        self.set_loading_state("Розрахунок матриці схожості...")
        try:
            matrices = self.extract_user_matrices(settings)
            D_total = sum(w * m for w, m in zip(weights, matrices))
            
            n_clust = self.u_n_clusters.value()
            algo = self.u_algo.currentText()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if algo == "Agglomerative (average)": clustering = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed', linkage='average')
                elif algo == "Agglomerative (complete)": clustering = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed', linkage='complete')
                elif algo == "Agglomerative (single)": clustering = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed', linkage='single')
                elif algo == "K-Medoids" and HAS_KMEDOIDS: clustering = KMedoids(n_clusters=n_clust, metric='precomputed')
                elif algo == "Spectral Clustering":
                    S = 1.0 - D_total
                    np.fill_diagonal(S, 1.0)
                    clustering = SpectralClustering(n_clusters=n_clust, affinity='precomputed')
                else: raise Exception("Невідомий алгоритм")
                
                matrix_for_clust = S if algo == "Spectral Clustering" else D_total
                self.current_users_clusters = clustering.fit_predict(matrix_for_clust)
            
            true_themes = []
            for item in self.users_orig_data:
                themes = item.get('Улюблені теми', [])
                true_themes.append(themes[0] if themes else "None")
                
            ari, ri = compute_rand_indices(true_themes, self.current_users_clusters)
            self.u_ari_label.setText(f"Adjusted Rand Index: {ari:.4f}")
            self.u_ri_label.setText(f"Rand Index: {ri:.4f}")

            self.u_color_mode.setCurrentIndex(0)
            self.draw_users_plot()
        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))
        finally:
            self.set_loading_state("Готово")

    def autotune_users(self):
        settings = self.u_weights_widget.get_settings()
        locked = [s['weight'] if s['locked'] else None for s in settings]
        self.set_loading_state("Попередня генерація матриць для оптимізації...")
        matrices = self.extract_user_matrices(settings)
        true_themes = []
        for item in self.users_orig_data:
            themes = item.get('Улюблені теми', [])
            true_themes.append(themes[0] if themes else "None")
        time_limit_sec = self.u_tune_time.value() * 60
        self.start_optimizer(matrices, locked, self.u_n_clusters.value(), self.u_algo.currentText(), true_themes, False, time_limit_sec)

    def draw_users_plot(self):
        if self.users_embedding is None: return
        self.fig_users.clear()
        self.ax_users = self.fig_users.add_subplot(111)
        mode = self.u_color_mode.currentText()
        self.ax_users.set_title(f"UMAP: Користувачі (Колір: {mode})")
        
        if mode == "Кластер":
            uniques, counts = np.unique(self.current_users_clusters, return_counts=True)
            n_clusters = len(uniques)
            if n_clusters > 0:
                mapped_clusters = np.array([np.where(uniques == c)[0][0] for c in self.current_users_clusters])
                cmap = plt.get_cmap('tab20', n_clusters)
                scatter = self.ax_users.scatter(
                    self.users_embedding[:, 0], self.users_embedding[:, 1], 
                    c=mapped_clusters, cmap=cmap, vmin=-0.5, vmax=n_clusters-0.5, s=40, alpha=0.7
                )
                self.cbar_users = self.fig_users.colorbar(scatter, ax=self.ax_users, ticks=range(n_clusters))
                self.cbar_users.set_ticklabels([f"Кл {int(c)} ({n})" for c, n in zip(uniques, counts)])
        else:
            is_list = mode == "Улюблені теми"
            c_arr, c_map = self.get_colors_and_legend(self.users_orig_data, mode, is_list)
            self.ax_users.scatter(self.users_embedding[:, 0], self.users_embedding[:, 1], c=c_arr, s=40, alpha=0.7)
            leg = [Patch(facecolor=c_map[k], label=k) for k in c_map]
            self.ax_users.legend(handles=leg, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
            
        self.fig_users.subplots_adjust(right=0.72 if mode != 'Кластер' else 0.8, left=0.1, top=0.9, bottom=0.1)
        
        self.annot_users = self.ax_users.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="white", alpha=0.9), zorder=1000, clip_on=False)
        self.annot_users.set_visible(False)
        self.hover_circle_users = Circle((0,0), self.u_hover_radius.value(), fill=False, color='red', linestyle='--', zorder=999)
        self.hover_circle_users.set_visible(False)
        self.ax_users.add_patch(self.hover_circle_users)
        self.canvas_users.draw()

    def hover_users(self, event):
        if event.inaxes == self.ax_users and self.users_embedding is not None:
            x, y = event.xdata, event.ydata
            radius = self.u_hover_radius.value()
            distances = np.sqrt((self.users_embedding[:, 0] - x)**2 + (self.users_embedding[:, 1] - y)**2)
            count = np.sum(distances <= radius)
            
            if count > 0:
                xlim = self.ax_users.get_xlim()
                if x > (xlim[0] + xlim[1]) / 2:
                    self.annot_users.set_position((-15, 15))
                    self.annot_users.set_horizontalalignment('right')
                else:
                    self.annot_users.set_position((15, 15))
                    self.annot_users.set_horizontalalignment('left')

                self.annot_users.xy = (x, y)
                self.annot_users.set_text(f"Юзерів у r={radius}: {count}")
                self.annot_users.set_visible(True)
                
                self.hover_circle_users.center = (x, y)
                self.hover_circle_users.set_radius(radius)
                self.hover_circle_users.set_visible(True)
                self.canvas_users.draw_idle()
            else:
                if self.annot_users.get_visible():
                    self.annot_users.set_visible(False)
                    self.hover_circle_users.set_visible(False)
                    self.canvas_users.draw_idle()
        else:
            if hasattr(self, 'annot_users') and self.annot_users.get_visible():
                self.annot_users.set_visible(False)
                self.hover_circle_users.set_visible(False)
                self.canvas_users.draw_idle()

    def sanitize_matrix(self, d_matrix):
        d_matrix = np.nan_to_num(d_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        np.fill_diagonal(d_matrix, 0.0)
        return (d_matrix + d_matrix.T) / 2.0

    def start_optimizer(self, matrices, locked_weights, n_clusters, algo_name, true_labels, is_course, max_time_sec):
        self.opt_dialog = QDialog(self)
        self.opt_dialog.setWindowTitle("Оптимізація ваг...")
        self.opt_dialog.resize(400, 150)
        
        layout = QVBoxLayout(self.opt_dialog)
        lbl = QLabel(f"Перебір комбінацій для максимізації Adjusted Rand Index...\nМакс. час: {max_time_sec // 60} хв.")
        layout.addWidget(lbl)
        
        pbar = QProgressBar()
        layout.addWidget(pbar)
        
        btn_stop = QPushButton("Зупинити достроково")
        layout.addWidget(btn_stop)
        
        self.opt_thread = OptimizerThread(matrices, locked_weights, 0.05, n_clusters, algo_name, true_labels, max_time_sec)
        btn_stop.clicked.connect(self.opt_thread.stop)
        
        def update_prog(cur, total):
            pbar.setMaximum(total)
            pbar.setValue(cur)
            
        def on_finished(best_w, h_ari, h_ri, h_iters):
            self.opt_dialog.accept()
            self.show_optimizer_results(best_w, h_ari, h_ri, h_iters, is_course)
            self.set_loading_state("Готово")
            
        self.opt_thread.progress.connect(update_prog)
        self.opt_thread.finished.connect(on_finished)
        self.opt_thread.start()
        self.opt_dialog.exec_()

    def show_optimizer_results(self, best_w, h_ari, h_ri, h_iters, is_course):
        if not h_ari:
            QMessageBox.information(self, "Результат", "Немає вільних ваг для оптимізації або час вийшов.")
            return
            
        res_dialog = QDialog(self)
        res_dialog.setWindowTitle("Результати оптимізації")
        res_dialog.resize(700, 500)
        layout = QVBoxLayout(res_dialog)
        
        fig = Figure(figsize=(7, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(h_iters, h_ari, label="Adjusted Rand", color='blue')
        ax.plot(h_iters, h_ri, label="Classic Rand", color='gray', alpha=0.5)
        ax.set_title("Зміна метрик під час перебору")
        ax.set_xlabel("Ітерація")
        ax.set_ylabel("Rand Score")
        ax.legend()
        layout.addWidget(canvas)
        
        info = QLabel(f"<b>Найкращий ARI:</b> {max(h_ari):.4f}<br><b>Знайдені ваги:</b> {best_w}")
        layout.addWidget(info)
        
        btn_apply = QPushButton("Застосувати ці ваги та перерахувати")
        btn_apply.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        def apply_and_close():
            if is_course:
                self.c_weights_widget.set_weights(best_w)
                self.process_courses()
            else:
                self.u_weights_widget.set_weights(best_w)
                self.process_users()
            res_dialog.accept()
            
        btn_apply.clicked.connect(apply_and_close)
        layout.addWidget(btn_apply)
        
        res_dialog.exec_()

    def get_colors_and_legend(self, data, key, is_list):
        unique_vals = set()
        for item in data:
            val = item.get(key)
            if is_list and isinstance(val, list):
                for v in val: unique_vals.add(str(v))
            elif not is_list and val is not None:
                unique_vals.add(str(val))
        unique_vals = sorted(list(unique_vals))
        cmap = plt.get_cmap('tab20')
        color_map = {val: cmap(i % 20)[:3] for i, val in enumerate(unique_vals)}
        
        colors = []
        for item in data:
            val = item.get(key)
            if is_list and isinstance(val, list) and len(val) > 0:
                valid_colors = [color_map[str(v)] for v in val if str(v) in color_map]
                if valid_colors: colors.append(np.mean(valid_colors, axis=0))
                else: colors.append((0.7, 0.7, 0.7))
            elif not is_list and val is not None and str(val) in color_map:
                colors.append(color_map[str(val)])
            else:
                colors.append((0.7, 0.7, 0.7))
        return np.array(colors), color_map

    def calculate_base_embeddings(self):
        self.set_loading_state("Генерація базових UMAP-мап для візуалізації...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.courses_data:
                df_c = pd.DataFrame(self.courses_data)
                t_cols = [c for c in df_c.columns if c.startswith('Тег_')]
                n_cols = ['Ціна_norm', 'Складність_norm', 'Тривалість_norm'] + [c for c in df_c.columns if c.startswith('ФорматКурсу_')]
                base_f = self.extract_and_pad_features(df_c, 'TF-IDF_vector', t_cols, n_cols)
                base_f = np.nan_to_num(base_f, nan=0.0)
                red = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
                self.courses_embedding = red.fit_transform(base_f)
                self.current_courses_clusters = np.zeros(len(self.courses_data))
                
            if self.users_data:
                df_u = pd.DataFrame(self.users_data)
                n_cols = ['Час_навчання_norm', 'Витрати_norm', 'Вік_norm', 'Освіта_norm'] + [c for c in df_u.columns if c.startswith('Мета навчання_')]
                t_cols = [c for c in df_u.columns if c.startswith('Пройдено_тег_') or c.startswith('Бажане_тег_') or c.startswith('Формат_')]
                base_f = self.extract_and_pad_features(df_u, 'dummy', t_cols, n_cols)
                base_f = np.nan_to_num(base_f, nan=0.0)
                red = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
                self.users_embedding = red.fit_transform(base_f)
                self.current_users_clusters = np.zeros(len(self.users_data))
            
        self.set_loading_state("Готово")

    def save_courses(self):
        if len(self.current_courses_clusters) == 0: return
        filename, ok = QInputDialog.getText(self, "Збереження", "Введіть назву файлу:")
        if ok and filename:
            if not filename.endswith('.json'): filename += '.json'
            filepath = os.path.join(self.saves_dir, filename)
            
            cluster_data = [{"Course_ID": row["Course_ID"], "Cluster_ID": int(self.current_courses_clusters[i])} for i, row in enumerate(self.courses_data)]
            settings = self.c_weights_widget.get_settings()
            algo = self.c_algo.currentText()
            
            out = {
                "metadata": {
                    "entity": "courses",
                    "algorithm": algo,
                    "features_config": settings
                },
                "clusters": cluster_data
            }
            
            with open(filepath, "w", encoding="utf-8") as f: json.dump(out, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Успіх", f"Збережено у:\n{filepath}")

    def load_courses_clusters(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Відкрити", self.saves_dir, "JSON (*.json)")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f: loaded = json.load(f)
                
                if isinstance(loaded, dict) and "clusters" in loaded:
                    cluster_data = loaded["clusters"]
                else:
                    cluster_data = loaded
                    
                c_map = {item['Course_ID']: item['Cluster_ID'] for item in cluster_data}
                self.current_courses_clusters = np.array([c_map.get(r['Course_ID'], -1) for r in self.courses_data])
                if self.courses_embedding is not None: self.draw_courses_plot()
            except Exception as e: QMessageBox.critical(self, "Помилка", str(e))

    def save_users(self):
        if len(self.current_users_clusters) == 0: return
        filename, ok = QInputDialog.getText(self, "Збереження", "Введіть назву файлу:")
        if ok and filename:
            if not filename.endswith('.json'): filename += '.json'
            filepath = os.path.join(self.saves_dir, filename)
            
            cluster_data = [{"User_ID": row["User_ID"], "Cluster_ID": int(self.current_users_clusters[i])} for i, row in enumerate(self.users_data)]
            settings = self.u_weights_widget.get_settings()
            algo = self.u_algo.currentText()
            
            out = {
                "metadata": {
                    "entity": "users",
                    "algorithm": algo,
                    "features_config": settings
                },
                "clusters": cluster_data
            }
            
            with open(filepath, "w", encoding="utf-8") as f: json.dump(out, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Успіх", f"Збережено у:\n{filepath}")

    def load_users_clusters(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Відкрити", self.saves_dir, "JSON (*.json)")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f: loaded = json.load(f)
                
                if isinstance(loaded, dict) and "clusters" in loaded:
                    cluster_data = loaded["clusters"]
                else:
                    cluster_data = loaded
                    
                u_map = {item['User_ID']: item['Cluster_ID'] for item in cluster_data}
                self.current_users_clusters = np.array([u_map.get(r['User_ID'], -1) for r in self.users_data])
                if self.users_embedding is not None: self.draw_users_plot()
            except Exception as e: QMessageBox.critical(self, "Помилка", str(e))

    def setup_data_view_tab(self):
        layout = QVBoxLayout(self.tab_data_view)
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Джерело даних:"))
        self.data_view_combo = QComboBox()
        self.data_view_combo.addItems(["Користувачі (Оригінал)", "Користувачі (Вектори)", "Курси (Оригінал)", "Курси (Вектори)"])
        control_layout.addWidget(self.data_view_combo)
        control_layout.addWidget(QLabel("Пошук по ID:"))
        self.search_id_input = QLineEdit()
        self.search_id_input.setFixedWidth(100)
        control_layout.addWidget(self.search_id_input)
        btn_show = QPushButton("Показати")
        btn_show.clicked.connect(self.display_data)
        control_layout.addWidget(btn_show)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        self.data_text_area = QTextEdit()
        self.data_text_area.setReadOnly(True)
        self.data_text_area.setStyleSheet("font-family: Consolas, Courier New; font-size: 13px;")
        layout.addWidget(self.data_text_area)

    def display_data(self):
        sel = self.data_view_combo.currentText()
        sid = self.search_id_input.text().strip()
        data = self.users_orig_data if "Користувачі" in sel and "Оригінал" in sel else \
               self.users_data if "Користувачі" in sel else \
               self.courses_orig_data if "Оригінал" in sel else self.courses_data
        field = "User_ID" if "Користувачі" in sel else "Course_ID"
        
        if not data:
            self.data_text_area.setText("Дані не знайдено.")
            return
        if not sid:
            txt = f"Відображення перших 3 записів з {len(data)}:\n\n"
            for i in range(min(3, len(data))): txt += json.dumps(data[i], ensure_ascii=False, indent=4) + "\n" + "="*60 + "\n\n"
            self.data_text_area.setText(txt)
        else:
            try: sid_val = float(sid) if 'Course' in sel and 'Вектори' in sel else int(sid)
            except: self.data_text_area.setText("Введіть числове ID."); return
            found = next((item for item in data if item.get(field) == sid_val or (isinstance(item.get(field), float) and int(item.get(field)) == int(sid_val))), None)
            self.data_text_area.setText(json.dumps(found, ensure_ascii=False, indent=4) if found else f"Об'єкт {sid} не знайдено.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = ClusteringPipelineApp()
    window.show()
    sys.exit(app.exec_())