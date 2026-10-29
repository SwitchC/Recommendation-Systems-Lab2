import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
import umap
from pathlib import Path
import warnings
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge, Patch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, 
                             QRadioButton, QButtonGroup, QTextEdit, QLabel, 
                             QMessageBox, QSplitter, QProgressBar, QDialog, QTabWidget, QComboBox, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

BASE_DIR = Path(__file__).parent

class RecommendationEngine:
    def __init__(self, users_data, courses_data, courses_clusters, comments_data, pool_limit=100):
        self.users = {u['User_ID']: u for u in users_data}
        self.courses = {c['Course_ID']: c for c in courses_data}
        self.unique_courses_list = list(self.courses.values())
        self.course_indices = {c['Course_ID']: i for i, c in enumerate(self.unique_courses_list)}
        self.course_ids = [c['Course_ID'] for c in self.unique_courses_list]
        self.clusters = {c['Course_ID']: c['Cluster_ID'] for c in courses_clusters.get('clusters', [])}
        self.features_config = courses_clusters.get('metadata', {}).get('features_config', [])
        self.comments = comments_data
        self.pool_limit = pool_limit
        
        self.D_total = None
        self.D_content = None
        self.user_histories = {}
        self.recommendations = []
        
        self.pool_stats = {
            "Strategy_A_Pool_1": {c: {"Активний": [], "Пасивний": [], "Новий": []} for c in range(10)},
            "Strategy_A_Pool_2": {c: {"Активний": [], "Пасивний": [], "Новий": []} for c in range(10)},
            "Strategy_B_Pool_3": {c: {"Активний": [], "Пасивний": [], "Новий": []} for c in range(10)},
            "Strategy_C_Pool_4": {c: {"Активний": [], "Пасивний": [], "Новий": []} for c in range(10)}
        }

    def _extract_ids(self, items):
        extracted = set()
        if not items: return extracted
        for item in items:
            if isinstance(item, dict) and 'Course_ID' in item: extracted.add(int(item['Course_ID']))
            elif isinstance(item, (int, float)): extracted.add(int(item))
            elif isinstance(item, str) and item.isdigit(): extracted.add(int(item))
        return extracted

    def extract_histories(self):
        comment_interactions = {}
        for comment in self.comments:
            uid = comment.get('User_ID')
            cid = comment.get('Course_ID')
            if uid and cid:
                if uid not in comment_interactions: comment_interactions[uid] = set()
                comment_interactions[uid].add(cid)

        for uid, user in self.users.items():
            history_dict = {}
            details = user.get('Деталізована історія взаємодій', {})
            interactions = [
                ('Кліки за курсами', 'Переглянутий'),
                ('Курси з коментарями', 'Прокоментований'),
                ('Історія оцінювання', 'Оцінений'),
                ('Додані в бажане', 'В бажаному'),
                ('Пройдені курси', 'Пройдений'),
                ('Куплені курси', 'Куплений')
            ]
            for cid in comment_interactions.get(uid, set()):
                if cid in self.courses: history_dict[cid] = 'Прокоментований'
            for key, itype in interactions:
                cids = self._extract_ids(details.get(key, []))
                for cid in cids:
                    if cid in self.courses: history_dict[cid] = itype
            self.user_histories[uid] = history_dict

    def build_distance_matrix(self):
        df = pd.DataFrame(self.unique_courses_list)
        N = len(df)
        self.D_total = np.zeros((N, N))
        
        for config in self.features_config:
            weight = config['weight']
            if weight <= 0: continue
            label, metric = config['label'], config['metric']
            d_mat = np.zeros((N, N))
            
            if "Текст" in label and 'TF-IDF_vector' in df.columns:
                raw = df['TF-IDF_vector'].tolist()
                maxlen = max((len(v) if isinstance(v, list) else 0) for v in raw)
                mat = np.array([(v + [0.0]*(maxlen-len(v)) if isinstance(v, list) else [0.0]*maxlen) for v in raw])
            elif "Теги" in label:
                tag_cols = [c for c in df.columns if c.startswith('Тег_')]
                mat = df[tag_cols].fillna(0).astype(bool).values if tag_cols else np.zeros((N, 0))
            elif "Формати" in label:
                fmt_cols = [c for c in df.columns if c.startswith('ФорматКурсу_')]
                mat = df[fmt_cols].fillna(0).values if fmt_cols else np.zeros((N, 0))
            elif "Числові" in label:
                num_cols = ['Ціна_norm', 'Складність_norm', 'Тривалість_norm']
                mat = df[[c for c in num_cols if c in df.columns]].fillna(0).values
            else: continue

            if mat.shape[1] > 0:
                if config.get('use_umap', False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        red = umap.UMAP(n_components=config.get('umap_dim', 10), metric=metric, random_state=42)
                        emb = red.fit_transform(mat)
                    d_mat = pairwise_distances(emb, metric='euclidean')
                else:
                    d_mat = pairwise_distances(mat, metric=metric)
                    if metric == 'cosine': d_mat = np.clip(d_mat, 0, 2) 
                    
                max_d = np.nanmax(d_mat)
                if max_d > 0: d_mat = d_mat / max_d
                
                if "Текст" in label:
                    self.D_content = np.nan_to_num(d_mat, nan=0.0).copy()
                    
            self.D_total += weight * np.nan_to_num(d_mat, nan=0.0)

    def _record_pool_stats(self, pool_cids, strategy_key, user_type):
        counts = {c: 0 for c in range(10)}
        for cid in pool_cids:
            clust = self.clusters.get(cid, -1)
            if 0 <= clust < 10: counts[clust] += 1
        for c in range(10):
            self.pool_stats[strategy_key][c][user_type].append(counts[c])

    def get_ranked_top10(self, pool_candidates, history_cids, limit=10):
        if not pool_candidates or not history_cids: return []
        hist_indices = [self.course_indices[cid] for cid in history_cids]
        scores = []
        for cid in pool_candidates:
            idx = self.course_indices[cid]
            avg_dist = np.mean(self.D_total[idx, hist_indices])
            scores.append((avg_dist, cid))
        scores.sort(key=lambda x: x[0])
        return [cid for dist, cid in scores[:limit]]

    def generate(self, progress_callback=None):
        self.extract_histories()
        self.build_distance_matrix()
        self.recommendations = []
        total_users = len(self.users)
        all_course_ids = set(self.course_ids)
        
        for i, (uid, user) in enumerate(self.users.items()):
            history_cids = set(self.user_histories.get(uid, {}).keys())
            user_type = user.get("User_Type", "Новий")
            rec_record = {
                "User_ID": uid, "User_Type": user_type,
                "Strategy_A_Pool_1_Top10": [], "Strategy_A_Pool_2_Top10": [],
                "Strategy_B_Pool_3_Top10": [], "Strategy_C_Pool_4_Top10": []
            }
            
            if history_cids:
                pool_1 = set()
                if self.D_content is not None:
                    for hcid in history_cids:
                        h_idx = self.course_indices.get(hcid)
                        if h_idx is not None:
                            dists = self.D_content[h_idx]
                            cands = []
                            for cid in all_course_ids:
                                if cid not in history_cids:
                                    c_idx = self.course_indices.get(cid)
                                    if c_idx is not None:
                                        cands.append((dists[c_idx], cid))
                            cands.sort(key=lambda x: x[0])
                            for _, cid in cands[:2]:
                                pool_1.add(cid)
                                
                self._record_pool_stats(pool_1, "Strategy_A_Pool_1", user_type)
                rec_record["Strategy_A_Pool_1_Top10"] = self.get_ranked_top10(pool_1, history_cids, limit=10)
                
                hist_clusters = [self.clusters.get(cid) for cid in history_cids if cid in self.clusters]
                if hist_clusters:
                    unique_hist_clusters = set(hist_clusters)
                    pool_2 = {cid for cid in all_course_ids if self.clusters.get(cid) in unique_hist_clusters} - history_cids
                    self._record_pool_stats(pool_2, "Strategy_A_Pool_2", user_type)
                    rec_record["Strategy_A_Pool_2_Top10"] = self.get_ranked_top10(pool_2, history_cids, limit=10)
                    
                    cluster_counts = pd.Series(hist_clusters).value_counts()
                    top_cluster = cluster_counts.index[0]
                    pool_3 = {cid for cid in all_course_ids if self.clusters.get(cid) == top_cluster} - history_cids
                    self._record_pool_stats(pool_3, "Strategy_B_Pool_3", user_type)
                    rec_record["Strategy_B_Pool_3_Top10"] = self.get_ranked_top10(pool_3, history_cids, limit=10)
                    
                    pool_4 = set()
                    for clust, prop in (cluster_counts / len(hist_clusters)).items():
                        target_size = int(round(prop * self.pool_limit))
                        clust_cands = {cid for cid in all_course_ids if self.clusters.get(cid) == clust} - history_cids
                        ranked = self.get_ranked_top10(clust_cands, history_cids, limit=target_size) 
                        pool_4.update(ranked)
                    
                    self._record_pool_stats(pool_4, "Strategy_C_Pool_4", user_type)
                    rec_record["Strategy_C_Pool_4_Top10"] = self.get_ranked_top10(pool_4, history_cids, limit=10)
            else:
                self._record_pool_stats(set(), "Strategy_A_Pool_1", user_type)
                self._record_pool_stats(set(), "Strategy_A_Pool_2", user_type)
                self._record_pool_stats(set(), "Strategy_B_Pool_3", user_type)
                self._record_pool_stats(set(), "Strategy_C_Pool_4", user_type)

            self.recommendations.append(rec_record)
            if progress_callback: progress_callback(int((i + 1) / total_users * 100))
        return self.recommendations


class GeneratorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, engine): super().__init__(); self.engine = engine
    def run(self):
        try:
            results = self.engine.generate(progress_callback=self.progress.emit)
            self.finished.emit(results)
        except Exception as e: self.error.emit(str(e))

# =====================================================================
# 2. ДІАЛОГ ДЛЯ ПЕРЕГЛЯДУ ТА ЗБЕРЕЖЕННЯ ЗВІТІВ (ОКРЕМІ SCATTER PIE ДЛЯ КОЖНОГО ПУЛУ)
# =====================================================================

class ReportsViewerDialog(QDialog):
    def __init__(self, engine, recommendations, courses_data, cluster_stats, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Попередній перегляд звітів (Графіки)")
        self.resize(1200, 800)
        self.engine = engine
        self.recommendations = recommendations
        self.courses_data = courses_data
        self.cluster_stats = cluster_stats
        
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 Зберегти всі графіки як PNG")
        self.btn_save.setStyleSheet("background-color: #2b5797; color: white; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_all_figures)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        
        self.figures = {}
        self.generate_all_reports()

    def generate_all_reports(self):
        strategies = [
            ("Strategy_A_Pool_1", "Strategy_A_Pool_1_Top10", "Пул №1 (Контентна схожість: Топ-2)", "Пул 1"),
            ("Strategy_A_Pool_2", "Strategy_A_Pool_2_Top10", "Пул №2 (Кластери з історії)", "Пул 2"),
            ("Strategy_B_Pool_3", "Strategy_B_Pool_3_Top10", "Пул №3 (Топ-кластер)", "Пул 3"),
            ("Strategy_C_Pool_4", "Strategy_C_Pool_4_Top10", "Пул №4 (Пропорційно)", "Пул 4")
        ]

        x_labels = []
        for i in range(10):
            if i in self.cluster_stats:
                total = self.cluster_stats[i]['total']
                top_tags = sorted(self.cluster_stats[i]['tags'].items(), key=lambda x: x[1], reverse=True)[:2]
                tag_str = "\n".join([f"{t[0][:15]} ({(t[1]/total*100):.0f}%)" for t in top_tags])
                x_labels.append(f"Кл {i} ({total} курсів)\n{tag_str}")
            else:
                x_labels.append(f"Кл {i}")

        users_types = ["Активний", "Пасивний", "Новий"]
        bar_colors = {
            "Активний": ['#ccebc5', '#7bccc4', '#2b8cbe'], 
            "Пасивний": ['#ffeda0', '#feb24c', '#f03b20'], 
            "Новий": ['#deebf7', '#9ecae1', '#3182bd']
        }
        pie_colors = {"Активний": "#4CAF50", "Пасивний": "#FF9800", "Новий": "#2196F3"}

        for strat_key, top10_key, strat_title, short_name in strategies:
            # -------------------------------------------------------------
            # 1. Стовпчаста діаграма для даної стратегії
            # -------------------------------------------------------------
            fig_bar = Figure(figsize=(12, 7))
            ax_bar = fig_bar.add_subplot(111)
            x = np.arange(10)
            width = 0.25
            legend_elements_bar = []

            for u_idx, u_type in enumerate(users_types):
                mins, avgs, maxs = [], [], []
                for c in range(10):
                    arr = self.engine.pool_stats[strat_key][c][u_type]
                    if not arr: arr = [0]
                    mins.append(np.min(arr))
                    avgs.append(np.mean(arr))
                    maxs.append(np.max(arr))
                    
                offset = (u_idx - 1) * width
                ax_bar.bar(x + offset, mins, width, color=bar_colors[u_type][0])
                ax_bar.bar(x + offset, avgs, width, bottom=mins, color=bar_colors[u_type][1])
                ax_bar.bar(x + offset, maxs, width, bottom=np.array(mins)+np.array(avgs), color=bar_colors[u_type][2])
                
                legend_elements_bar.append(Patch(facecolor=bar_colors[u_type][0], label=f'{u_type} (Min)'))
                legend_elements_bar.append(Patch(facecolor=bar_colors[u_type][1], label=f'{u_type} (Avg)'))
                legend_elements_bar.append(Patch(facecolor=bar_colors[u_type][2], label=f'{u_type} (Max)'))
                
                for i in range(10):
                    if maxs[i] == 0 and avgs[i] == 0 and mins[i] == 0:
                        ax_bar.text(x[i]+offset, 0.5, "0", ha='center', va='bottom', fontsize=7, color='gray')
                    else:
                        if mins[i] > 0: ax_bar.text(x[i]+offset, mins[i]/2, f"{mins[i]}", ha='center', va='center', fontsize=7)
                        if avgs[i] > 0: ax_bar.text(x[i]+offset, mins[i]+avgs[i]/2, f"{avgs[i]:.1f}", ha='center', va='center', fontsize=7)
                        if maxs[i] > 0: ax_bar.text(x[i]+offset, mins[i]+avgs[i]+maxs[i]/2, f"{maxs[i]}", ha='center', va='center', fontsize=7)

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax_bar.set_title(f"{strat_title}: Аналіз розмірів пулу (Min, Avg, Max)")
            
            if strat_key == "Strategy_C_Pool_4":
                ax_bar.axhline(y=self.engine.pool_limit, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                legend_elements_bar.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=f'Ліміт пулу ({self.engine.pool_limit})'))
            
            ax_bar.legend(handles=legend_elements_bar, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
            fig_bar.tight_layout()
            
            self.figures[f"bar_chart_{strat_key}.png"] = fig_bar
            self._add_tab(fig_bar, f"{short_name} (Стовпчики)")

            # -------------------------------------------------------------
            # 2. Бульбашкова діаграма (Scatter Pie) ОКРЕМО для кожної стратегії
            # -------------------------------------------------------------
            fig_pie = Figure(figsize=(12, 10))
            ax_pie = fig_pie.add_subplot(111)

            cluster_data = {i: {"Активний": 0, "Пасивний": 0, "Новий": 0} for i in range(10)}
            
            # Підраховуємо ТОП-10 саме для поточної стратегії (top10_key)
            for rec in self.recommendations:
                u_type = rec.get("User_Type", "Новий")
                if u_type not in ["Активний", "Пасивний", "Новий"]: u_type = "Новий"
                
                for cid in rec.get(top10_key, []):
                    clust = self.engine.clusters.get(cid, -1)
                    if 0 <= clust < 10:
                        cluster_data[clust][u_type] += 1
                        
            circle_r = 25
            angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
            centers_x = np.cos(angles) * circle_r
            centers_y = np.sin(angles) * circle_r
            
            max_count = max(sum(counts.values()) for counts in cluster_data.values()) if cluster_data else 1
            if max_count == 0: max_count = 1
            
            for i in range(10):
                counts = cluster_data[i]
                total = sum(counts.values())
                
                clust_total_courses = self.cluster_stats.get(i, {}).get('total', 0)
                top_tags = sorted(self.cluster_stats.get(i, {}).get('tags', {}).items(), key=lambda x: x[1], reverse=True)[:2]
                
                tag_str = ""
                for tag, count in top_tags:
                    pct = (count / clust_total_courses * 100) if clust_total_courses > 0 else 0
                    tag_str += f"{tag[:15]} ({pct:.0f}%)\n"

                if total == 0:
                    # Якщо в кластер нічого не потрапило
                    ax_pie.plot(centers_x[i], centers_y[i], 'o', color='lightgray', markersize=6)
                    ax_pie.text(centers_x[i], centers_y[i] - 2.0, 
                                f"Кластер {i} (Курсів: {clust_total_courses})\nРекомендовано: 0 разів\n{tag_str.strip()}", 
                                ha='center', va='top', fontsize=8, color='gray')
                    continue
                
                radius = (np.sqrt(total / max_count)) * 6.0 
                start_angle = 0
                for u_type, count in counts.items():
                    if count > 0:
                        theta = 360 * (count / total)
                        wedge = Wedge((centers_x[i], centers_y[i]), radius, start_angle, start_angle + theta, 
                                      facecolor=pie_colors[u_type], edgecolor='white', alpha=0.85)
                        ax_pie.add_patch(wedge)
                        start_angle += theta
                
                ax_pie.text(centers_x[i], centers_y[i] - radius - 1.0, 
                            f"Кластер {i} (Курсів: {clust_total_courses})\nРекомендовано: {total} разів\n{tag_str.strip()}", 
                            ha='center', va='top', fontsize=8, fontweight='bold', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            legend_elements_pie = [Patch(facecolor=pie_colors[k], label=k) for k in pie_colors.keys()]
            ax_pie.legend(handles=legend_elements_pie, loc='upper right', title="Тип користувача", fontsize=9)
            
            ax_pie.set_xlim(-35, 35)
            ax_pie.set_ylim(-35, 35)
            ax_pie.axis('off')
            ax_pie.set_title(f"{strat_title}: Розподіл ТОП-10 рекомендацій по кластерам (Scatter Pie)", fontsize=13, pad=20)
            
            fig_pie.tight_layout()
            self.figures[f"scatter_pie_{strat_key}.png"] = fig_pie
            self._add_tab(fig_pie, f"{short_name} (Scatter Pie)")

    def _add_tab(self, fig, title):
        tab = QWidget()
        t_layout = QVBoxLayout(tab)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, tab)
        t_layout.addWidget(toolbar)
        t_layout.addWidget(canvas)
        self.tabs.addTab(tab, title)

    def save_all_figures(self):
        reports_dir = BASE_DIR / "reports"
        reports_dir.mkdir(exist_ok=True)
        for name, fig in self.figures.items():
            fig.savefig(reports_dir / name, dpi=150)
        QMessageBox.information(self, "Звіти згенеровано", f"Усі 8 графіків успішно збережено у папку:\n{reports_dir.absolute()}")

    def generate_scatter_pie(self):
        fig = Figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        cluster_data = {i: {"Активний": 0, "Пасивний": 0, "Новий": 0} for i in range(10)}
        
        for rec in self.recommendations:
            u_type = rec.get("User_Type", "Новий")
            if u_type not in ["Активний", "Пасивний", "Новий"]: u_type = "Новий"
            
            for cid in rec.get("Strategy_A_Pool_1_Top10", []):
                clust = self.engine.clusters.get(cid, -1)
                if 0 <= clust < 10:
                    cluster_data[clust][u_type] += 1
                    
        circle_r = 25
        angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
        centers_x = np.cos(angles) * circle_r
        centers_y = np.sin(angles) * circle_r
        
        colors = {"Активний": "#4CAF50", "Пасивний": "#FF9800", "Новий": "#2196F3"}
        max_count = max(sum(counts.values()) for counts in cluster_data.values()) if cluster_data else 1
        
        for i in range(10):
            counts = cluster_data[i]
            total = sum(counts.values())
            if total == 0: continue
            
            radius = (np.sqrt(total / max_count)) * 6.0 
            
            start_angle = 0
            for u_type, count in counts.items():
                if count > 0:
                    theta = 360 * (count / total)
                    wedge = Wedge((centers_x[i], centers_y[i]), radius, start_angle, start_angle + theta, facecolor=colors[u_type], edgecolor='white', alpha=0.85)
                    ax.add_patch(wedge)
                    start_angle += theta
            
            clust_total_courses = self.cluster_stats.get(i, {}).get('total', 0)
            top_tags = sorted(self.cluster_stats.get(i, {}).get('tags', {}).items(), key=lambda x: x[1], reverse=True)[:2]
            
            tag_str = ""
            for tag, count in top_tags:
                pct = (count / clust_total_courses * 100) if clust_total_courses > 0 else 0
                tag_str += f"{tag[:15]} ({pct:.0f}%)\n"
            
            ax.text(centers_x[i], centers_y[i] - radius - 1.0, 
                    f"Кластер {i} (Курсів: {clust_total_courses})\nРекомендовано: {total} разів\n{tag_str.strip()}", 
                    ha='center', va='top', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors.keys()]
        ax.legend(handles=legend_elements, loc='upper right', title="Тип користувача", fontsize=9)
        
        ax.set_xlim(-35, 35)
        ax.set_ylim(-35, 35)
        ax.axis('off')
        ax.set_title("Розподіл фінальних ТОП-10 рекомендацій по кластерам (Scatter Pie)", fontsize=14, pad=20)
        
        fig.tight_layout()
        self.figures["scatter_pie_top10.png"] = fig
        
        tab = QWidget()
        t_layout = QVBoxLayout(tab)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, tab)
        t_layout.addWidget(toolbar)
        t_layout.addWidget(canvas)
        self.tabs.addTab(tab, "Scatter Pie (ТОП-10)")

    def save_all_figures(self):
        reports_dir = BASE_DIR / "reports"
        reports_dir.mkdir(exist_ok=True)
        for name, fig in self.figures.items():
            fig.savefig(reports_dir / name, dpi=150)
        QMessageBox.information(self, "Звіти згенеровано", f"Усі графіки збережено у папку:\n{reports_dir.absolute()}")


class RecommendationsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор Рекомендацій ЛР №2 (Force-Directed Graph & Reports)")
        self.resize(1600, 900)
        
        self.users_data = []
        self.courses_data = []
        self.courses_clusters = {}
        self.users_clusters = {}
        self.comments_data = []
        self.recommendations = []
        self.user_accuracies = {}
        self.cluster_stats_cache = {}
        
        self.current_user_id = None
        self.current_pool_key = "Strategy_A_Pool_1_Top10"
        
        self.G = None
        self.pos = {}
        self.dragging_node = None
        
        self.node_colors = []
        self.node_labels = {}
        self.hist_edges = []
        self.rec_edges = []
        self.edge_weights = []
        self.edge_labels = {}
        
        self.type_colors = {
            'Куплений': '#2ca02c', 'Пройдений': '#9467bd',
            'В бажаному': '#e377c2', 'Оцінений': '#8c564b',
            'Прокоментований': '#7f7f7f', 'Переглянутий': '#bcbd22'
        }
        
        self.init_ui()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        top_panel = QHBoxLayout()
        self.btn_load_data = QPushButton("1. Завантажити дані")
        self.btn_load_data.clicked.connect(self.load_all_data)
        self.btn_generate = QPushButton("2. Згенерувати рекомендації")
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self.run_generation)
        self.btn_save = QPushButton("3. Зберегти .json")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_recs)
        self.btn_load_recs = QPushButton("Відкрити існуючий .json")
        self.btn_load_recs.clicked.connect(self.load_recs)
        self.btn_reports = QPushButton("📊 Згенерувати графіки звітів")
        self.btn_reports.setEnabled(False)
        self.btn_reports.setStyleSheet("background-color: #d35400; color: white; font-weight: bold;")
        self.btn_reports.clicked.connect(self.open_reports_dialog)
        
        top_panel.addWidget(self.btn_load_data)
        top_panel.addWidget(self.btn_generate)
        top_panel.addWidget(self.btn_save)
        top_panel.addWidget(self.btn_load_recs)
        top_panel.addWidget(self.btn_reports)
        layout.addLayout(top_panel)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        strat_panel = QHBoxLayout()
        strat_panel.addWidget(QLabel("<b>Вибір пулу:</b>"))
        self.radio_group = QButtonGroup(self)
        r1 = QRadioButton("Пул №1")
        r1.setChecked(True)
        r2 = QRadioButton("Пул №2")
        r3 = QRadioButton("Пул №3")
        r4 = QRadioButton("Пул №4")
        
        for i, r in enumerate([r1, r2, r3, r4], 1):
            self.radio_group.addButton(r, i)
            strat_panel.addWidget(r)
        self.radio_group.buttonClicked.connect(self.on_pool_changed)
        
        strat_panel.addSpacing(15)
        strat_panel.addWidget(QLabel("<b>Ліміт Пулу №4:</b>"))
        self.spin_pool_limit = QSpinBox()
        self.spin_pool_limit.setRange(10, 1000)
        self.spin_pool_limit.setValue(100)
        self.spin_pool_limit.setSingleStep(10)
        strat_panel.addWidget(self.spin_pool_limit)
        
        self.cb_hide_unconnected = QCheckBox("Сховати історію без зв'язків")
        self.cb_hide_unconnected.setStyleSheet("font-weight: bold; color: #2b5797;")
        self.cb_hide_unconnected.toggled.connect(self.build_graph)
        strat_panel.addSpacing(15)
        strat_panel.addWidget(self.cb_hide_unconnected)
        
        layout.addLayout(strat_panel)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Сортування:"))
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["ID (зростання)", "Точність (спадання)", "Точність (зростання)"])
        self.combo_sort.currentIndexChanged.connect(self.populate_user_list)
        sort_layout.addWidget(self.combo_sort)
        left_layout.addLayout(sort_layout)
        self.lbl_global_stats = QLabel("Середня точність: N/A\nМін: N/A | Макс: N/A")
        self.lbl_global_stats.setStyleSheet("font-size: 12px; font-weight: bold; color: #333; margin-bottom: 5px;")
        left_layout.addWidget(self.lbl_global_stats)
        self.user_list = QListWidget()
        self.user_list.itemClicked.connect(self.on_user_selected)
        left_layout.addWidget(self.user_list)
        splitter.addWidget(left_panel)
        
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        graph_layout.addWidget(self.toolbar)
        graph_layout.addWidget(self.canvas)
        splitter.addWidget(graph_widget)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_metric = QLabel("Точність влучання: N/A")
        self.lbl_metric.setStyleSheet("font-size: 14px; font-weight: bold; color: #2b5797;")
        right_layout.addWidget(self.lbl_metric)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("font-family: Consolas;")
        right_layout.addWidget(self.info_text)
        splitter.addWidget(right_panel)
        
        splitter.setSizes([250, 900, 350])

    def load_all_data(self):
        try:
            with open(BASE_DIR / "original/users_raw.json", "r", encoding="utf-8") as f: self.users_data = json.load(f)
            with open(BASE_DIR / "original/courses_raw.json", "r", encoding="utf-8") as f: self.courses_data = json.load(f)
            with open(BASE_DIR / "original/comments_final.json", "r", encoding="utf-8") as f: comments_1 = json.load(f)
            with open(BASE_DIR / "original/comments_queue.json", "r", encoding="utf-8") as f: comments_2 = json.load(f)
            self.comments_data = comments_1 + comments_2
            
            vec_courses_path = BASE_DIR / "vectorized/courses_vectorized.json"
            if vec_courses_path.exists():
                with open(vec_courses_path, "r", encoding="utf-8") as f:
                    courses_vec = json.load(f)
                    vec_dict = {item.get('Course_ID', item.get('id')): item for item in courses_vec}
                    for course in self.courses_data:
                        cid = course.get('Course_ID', course.get('id'))
                        if cid in vec_dict: course.update(vec_dict[cid])

            courses_cluster_path = BASE_DIR / "saved_clusterings/courses_k10.json"
            if courses_cluster_path.exists():
                with open(courses_cluster_path, "r", encoding="utf-8") as f: self.courses_clusters = json.load(f)
            
            self.build_cluster_stats()
            self.populate_user_list()
            self.btn_generate.setEnabled(True)
            QMessageBox.information(self, "Успіх", "Усі дані автоматично завантажено!")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити дані:\n{str(e)}")

    def build_cluster_stats(self):
        self.cluster_stats_cache = {}
        if not self.courses_clusters: return
        cluster_list = self.courses_clusters.get('clusters', [])
        
        c_by_clust = {}
        for item in cluster_list:
            cid = item['Course_ID']
            cl = item['Cluster_ID']
            if cl not in c_by_clust: c_by_clust[cl] = []
            c_by_clust[cl].append(cid)
            
        for cl, cids in c_by_clust.items():
            total = len(cids)
            tag_counts = {}
            for cid in cids:
                c_data = next((c for c in self.courses_data if c['Course_ID'] == cid), {})
                tags = c_data.get('Теги', [])
                if tags:
                    t = tags[0]
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            self.cluster_stats_cache[cl] = {'total': total, 'tags': tag_counts}

    def calculate_all_accuracies(self):
        self.user_accuracies = {}
        for rec in self.recommendations:
            uid = rec['User_ID']
            top10 = rec.get(self.current_pool_key, [])
            user = next((u for u in self.users_data if u['User_ID'] == uid), None)
            
            if not user or not top10:
                self.user_accuracies[uid] = 0.0
                continue
                
            truth_themes = user.get('Улюблені теми', [])
            truth_weights = user.get('Ваги тем', [])
            if not truth_themes:
                self.user_accuracies[uid] = 0.0
                continue
            
            rec_tags = []
            for cid in top10:
                c_data = next((c for c in self.courses_data if c['Course_ID'] == cid), {})
                rtags = c_data.get('Теги', [])
                if isinstance(rtags, list):
                    rec_tags.extend(rtags)
                    
            vocab = list(set(truth_themes + rec_tags))
            vocab.sort()
            
            vec_truth = np.zeros(len(vocab), dtype=float)
            vec_rec = np.zeros(len(vocab), dtype=float)
            
            for i, tag in enumerate(vocab):
                if tag in truth_themes:
                    idx = truth_themes.index(tag)
                    vec_truth[i] = truth_weights[idx] if idx < len(truth_weights) else 0.0
                vec_rec[i] = rec_tags.count(tag)
                
            sum_rec = np.sum(vec_rec)
            if sum_rec > 0: vec_rec = vec_rec / sum_rec
            
            if np.linalg.norm(vec_truth) == 0 or np.linalg.norm(vec_rec) == 0:
                self.user_accuracies[uid] = 0.0
            else:
                self.user_accuracies[uid] = cosine_similarity([vec_truth], [vec_rec])[0][0] * 100

        accuracies = [a for a in self.user_accuracies.values() if a > 0.0]
        if accuracies:
            self.lbl_global_stats.setText(f"Середня точність: {np.mean(accuracies):.1f}%\nМін: {np.min(accuracies):.1f}% | Макс: {np.max(accuracies):.1f}%")
        else:
            self.lbl_global_stats.setText("Середня точність: 0.0%\nМін: 0.0% | Макс: 0.0%")

    def populate_user_list(self):
        self.user_list.clear()
        unique_users = list({u.get('User_ID', u.get('id')): u for u in self.users_data}.values())
        
        sort_mode = self.combo_sort.currentIndex()
        if sort_mode == 1: unique_users.sort(key=lambda u: self.user_accuracies.get(u['User_ID'], 0.0), reverse=True)
        elif sort_mode == 2: unique_users.sort(key=lambda u: self.user_accuracies.get(u['User_ID'], 0.0), reverse=False)
        else: unique_users.sort(key=lambda u: u['User_ID'])
            
        for u in unique_users:
            uid = u['User_ID']
            acc = self.user_accuracies.get(uid, 0.0)
            acc_str = f" [{acc:.1f}%]" if self.recommendations else ""
            item = QListWidgetItem(f"User {uid}{acc_str}")
            item.setData(Qt.UserRole, uid)
            utype = u.get("User_Type", "")
            if utype == "Активний": item.setBackground(QColor("#d4edda")) 
            elif utype == "Пасивний": item.setBackground(QColor("#fff3cd"))
            elif utype == "Новий": item.setBackground(QColor("#d1ecf1"))
            self.user_list.addItem(item)

    def run_generation(self):
        self.btn_generate.setEnabled(False)
        self.progress_bar.setValue(0)
        
        pool_lim = self.spin_pool_limit.value()
        self.engine = RecommendationEngine(self.users_data, self.courses_data, self.courses_clusters, self.comments_data, pool_limit=pool_lim)
        
        self.thread = GeneratorThread(self.engine)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_generation_finished)
        self.thread.error.connect(lambda e: QMessageBox.critical(self, "Помилка", e))
        self.thread.start()

    def on_generation_finished(self, results):
        self.recommendations = results
        self.btn_save.setEnabled(True)
        self.btn_reports.setEnabled(True)
        self.progress_bar.setValue(100)
        self.calculate_all_accuracies()
        self.populate_user_list()
        QMessageBox.information(self, "Готово", "Генерацію та ранжування завершено!")
        if self.current_user_id: self.build_graph()

    def save_recs(self):
        filepath = BASE_DIR / "recommendations.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f: json.dump(self.recommendations, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Успіх", f"Збережено:\n{filepath.name}")
        except Exception as e: QMessageBox.critical(self, "Помилка", str(e))

    def load_recs(self):
        filepath = BASE_DIR / "recommendations.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f: self.recommendations = json.load(f)
                self.btn_save.setEnabled(True)
                self.btn_reports.setEnabled(True)
                
                if not hasattr(self, 'engine') or self.engine is None:
                    self.engine = RecommendationEngine(self.users_data, self.courses_data, self.courses_clusters, self.comments_data, pool_limit=self.spin_pool_limit.value())
                    self.engine.extract_histories()
                    self.engine.build_distance_matrix()
                elif not self.engine.user_histories:
                    self.engine.extract_histories()
                    self.engine.build_distance_matrix()
                    
                if not self.engine.pool_stats["Strategy_A_Pool_1"][0]["Активний"]:
                    for rec in self.recommendations:
                        ut = rec.get("User_Type", "Новий")
                        for k in ["Strategy_A_Pool_1", "Strategy_A_Pool_2", "Strategy_B_Pool_3", "Strategy_C_Pool_4"]:
                            pool_key = k + "_Top10"
                            self.engine._record_pool_stats(rec.get(pool_key, []), k, ut)
                    
                self.calculate_all_accuracies()
                self.populate_user_list()
                if self.current_user_id: self.build_graph()
                QMessageBox.information(self, "Успіх", f"Завантажено:\n{filepath.name}")
            except Exception as e: QMessageBox.critical(self, "Помилка", str(e))
        else: QMessageBox.warning(self, "Увага", "Файл recommendations.json відсутній.")

    def on_pool_changed(self, btn):
        mapping = {1: "Strategy_A_Pool_1_Top10", 2: "Strategy_A_Pool_2_Top10", 3: "Strategy_B_Pool_3_Top10", 4: "Strategy_C_Pool_4_Top10"}
        self.current_pool_key = mapping[self.radio_group.checkedId()]
        if self.recommendations:
            self.calculate_all_accuracies()
            self.populate_user_list()
        if self.current_user_id: self.build_graph()

    def on_user_selected(self, item):
        self.current_user_id = item.data(Qt.UserRole)
        self.build_graph()
        self.show_user_info(self.current_user_id)

    def open_reports_dialog(self):
        dialog = ReportsViewerDialog(self.engine, self.recommendations, self.courses_data, self.cluster_stats_cache, self)
        dialog.exec_()

    def build_graph(self):
        self.ax.clear()
        
        if not self.recommendations or not self.current_user_id: return
        user_rec = next((r for r in self.recommendations if r["User_ID"] == self.current_user_id), None)
        if not user_rec: return
        
        top10 = user_rec.get(self.current_pool_key, [])
        if not top10:
            self.ax.text(0.5, 0.5, "Пул порожній: Cold Start", ha='center', va='center', fontsize=12, color='red')
            self.ax.axis('off')
            self.lbl_metric.setText("Точність влучання: 0.0% (Немає даних)")
            self.canvas.draw()
            return
            
        history_dict = self.engine.user_histories.get(self.current_user_id, {}) if hasattr(self, 'engine') else {}
        hist_cids = list(history_dict.keys())

        self.G = nx.Graph()
        user_node = f"U{self.current_user_id}"
        self.G.add_node(user_node, type='user')
        
        for cid in hist_cids:
            self.G.add_node(cid, type='history', i_type=history_dict[cid])
            self.G.add_edge(user_node, cid, weight=0.0, label="")
            
        connected_hist_nodes = set()
        
        for cid in top10:
            self.G.add_node(cid, type='rec')
            c_idx = self.engine.course_indices.get(cid)
            hist_indices = [self.engine.course_indices[h] for h in hist_cids if h in self.engine.course_indices]
            
            avg_dist = 0.0
            if c_idx is not None and hist_indices:
                sub_dists = self.engine.D_total[c_idx, hist_indices]
                avg_dist = np.mean(sub_dists)
                best_idx = np.argmin(sub_dists)
                min_dist = sub_dists[best_idx]
                closest_hist_cid = hist_cids[best_idx]
                self.G.add_edge(closest_hist_cid, cid, weight=min_dist, label=f"{min_dist:.2f}")
                connected_hist_nodes.add(closest_hist_cid)
            else:
                self.G.add_edge(hist_cids[0] if hist_cids else user_node, cid, weight=0.5, label="N/A")
                if hist_cids: connected_hist_nodes.add(hist_cids[0])
                
            self.G.nodes[cid]['avg_dist'] = avg_dist

        if self.cb_hide_unconnected.isChecked():
            to_remove = [n for n in hist_cids if n not in connected_hist_nodes]
            self.G.remove_nodes_from(to_remove)

        initial_pos = nx.random_layout(self.G, seed=42)
        initial_pos[user_node] = np.array([0.0, 0.0])
        self.pos = nx.spring_layout(self.G, pos=initial_pos, fixed=[user_node], k=3.0, iterations=250, seed=42, scale=4.0)
        
        self.node_colors, self.node_labels = [], {}
        user = next((u for u in self.users_data if u['User_ID'] == self.current_user_id), {})
        
        for node in self.G.nodes():
            ntype = self.G.nodes[node]['type']
            if ntype == 'user':
                self.node_colors.append('#1f77b4')
                themes = user.get('Улюблені теми', [])
                th_str = "\n".join(themes[:2]) + (".." if len(themes)>2 else "")
                self.node_labels[node] = f"{node}\n{th_str}"
            else:
                c_data = next((c for c in self.courses_data if c['Course_ID'] == node), {})
                tags = c_data.get('Теги', [])
                tag_name = tags[0] if tags else "Unknown"
                
                cl_id = next((item['Cluster_ID'] for item in self.courses_clusters.get('clusters', []) if item['Course_ID'] == node), -1)
                pct = 0.0
                if cl_id in self.cluster_stats_cache:
                    total_in_cl = self.cluster_stats_cache[cl_id]['total']
                    tag_count = self.cluster_stats_cache[cl_id]['tags'].get(tag_name, 0)
                    if total_in_cl > 0: pct = (tag_count / total_in_cl) * 100
                    
                tag_str = f"{tag_name[:12]} ({pct:.0f}%)"
                
                if ntype == 'history':
                    i_type = self.G.nodes[node]['i_type']
                    self.node_colors.append(self.type_colors.get(i_type, '#000000'))
                    self.node_labels[node] = f"ID:{node}\n{tag_str}"
                else:
                    self.node_colors.append('#ff7f0e')
                    avg = self.G.nodes[node]['avg_dist']
                    self.node_labels[node] = f"ID:{node}\n{tag_str}\nAvg:{avg:.2f}"

        self.hist_edges = [(u, v) for u, v in self.G.edges() if self.G.nodes[u]['type'] == 'user' or self.G.nodes[v]['type'] == 'user']
        self.rec_edges = [(u, v) for u, v in self.G.edges() if (u, v) not in self.hist_edges]
        self.edge_weights = [self.G[u][v]['weight'] for u, v in self.rec_edges]
        self.edge_labels = {(u, v): self.G[u][v]['label'] for u, v in self.rec_edges}

        acc = self.user_accuracies.get(self.current_user_id, 0.0)
        self.lbl_metric.setText(f"Точність влучання (Cosine Similarity): {acc:.1f}%")
        self.render_graph()

    def render_graph(self):
        if not self.G or not self.pos: return
        xlim, ylim = None, None
        if self.ax.has_data() and self.dragging_node:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        self.ax.clear()
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edgelist=self.hist_edges, edge_color='gray', width=1.0, alpha=0.2)
        
        if self.rec_edges:
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edgelist=self.rec_edges, 
                                   edge_color=self.edge_weights, edge_cmap=plt.cm.RdYlGn_r, 
                                   edge_vmin=0.0, edge_vmax=1.0, width=2.5, alpha=0.85)
            nx.draw_networkx_edge_labels(self.G, self.pos, ax=self.ax, edge_labels=self.edge_labels, font_size=7, 
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))

        nodes = nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=self.node_colors, 
                               node_size=1000, alpha=0.9, edgecolors='white', linewidths=1.0)
        
        nx.draw_networkx_labels(self.G, self.pos, self.node_labels, font_size=7, font_color='black', ax=self.ax,
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver', boxstyle='round,pad=0.3'))
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='User', markerfacecolor='#1f77b4', markersize=9),
            Line2D([0], [0], marker='o', color='w', label='Рекомендація', markerfacecolor='#ff7f0e', markersize=9)
        ]
        for name, col in self.type_colors.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{name}', markerfacecolor=col, markersize=9))
            
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=8, bbox_to_anchor=(1.0, 1.0))
        self.ax.axis('off')
        
        if xlim and ylim and self.dragging_node:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            
        self.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        scale = 1.1 if event.button == 'down' else 1/1.1
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_width, new_height = (xlim[1] - xlim[0]) * scale, (ylim[1] - ylim[0]) * scale
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if self.toolbar.mode != '': return 
        if not hasattr(self, 'pos') or not self.pos: return
        
        x, y = event.xdata, event.ydata
        closest, min_dist = None, float('inf')
        for n, p in self.pos.items():
            d = (p[0] - x)**2 + (p[1] - y)**2
            if d < min_dist: min_dist, closest = d, n
                
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        threshold = ((xlim[1] - xlim[0]) * 0.03)**2 + ((ylim[1] - ylim[0]) * 0.03)**2
        
        if min_dist < threshold:
            self.dragging_node = closest
            ntype = self.G.nodes[closest]['type']
            if ntype == 'user': self.show_user_info(int(str(closest)[1:]))
            else: self.show_course_info(int(closest))

    def on_release(self, event):
        self.dragging_node = None

    def on_motion(self, event):
        if self.dragging_node is None or event.inaxes != self.ax: return
        self.pos[self.dragging_node] = np.array([event.xdata, event.ydata])
        self.render_graph()

    def show_user_info(self, uid):
        user = next((u for u in self.users_data if u['User_ID'] == uid), None)
        if user:
            display_u = {k:v for k,v in user.items() if k not in ['Загальний час навчання по місяцях', 'Загальні витрати по місяцях', 'Пройдені курси за тегами']}
            self.info_text.setText(json.dumps(display_u, ensure_ascii=False, indent=4))

    def show_course_info(self, cid):
        course = next((c for c in self.courses_data if c['Course_ID'] == cid), None)
        if course:
            display_c = {k:v for k,v in course.items() if k != 'Деталізована історія взаємодій'}
            self.info_text.setText(json.dumps(display_c, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RecommendationsApp()
    window.show()
    sys.exit(app.exec_())