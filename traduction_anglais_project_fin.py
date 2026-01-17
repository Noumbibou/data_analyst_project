import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from scipy.stats import chi2_contingency
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'primary': '#1a237e',
    'secondary': '#0288d1',
    'accent': '#ff6f00',
    'success': '#2e7d32',
    'danger': '#c62828',
    'warning': '#f57c00',
    'info': '#17a2b8',

    'bg_main': '#f5f5f5',
    'bg_card': '#ffffff',
    'bg_dark': '#263238',

    'text_primary': '#212121',
    'text_secondary': '#757575',
    'text_white': '#ffffff',

    'gradient_start': '#1565c0',
    'gradient_end': '#0277bd',
    'shadow': '#9e9e9e'
}

CONSTRAINTS = {
    'Books': (100000, 10000000),
    'Seats': (10, 5000),
    'Visitors_per_day': (10, 20000),
    'Temperature_C': (18, 24),
    'Humidity_percent': (30, 60),
    'Staff': (1, 500),
    'Budget_k_euros': (10, 50000),
    'Opening_hours_per_day': (4, 24)
}

class AppData:
    def __init__(self):
        self.main_data = None
        self.scaled_data = None
        self.pca = None
        self.individuals_coords_df = None
        self.variables_coords_df = None
        self.explained_variance = None

        self.rf_model = None
        self.feature_cols = None
        self.kmeans_model = None
        self.n_clusters = None
        self.cluster_labels = None

        self.contingency_table = None

app_data = AppData()

def clean_numeric_dataframe(df):
    """Clean and convert columns to numeric"""
    df_clean = df.copy()
    numeric_cols = []
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str).str.replace(" ", "").str.replace(",", ".")
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            numeric_cols.append(col)
    return df_clean[numeric_cols]

def show_dataframe_window(title, df):
    """Display a DataFrame in a modern window"""
    window = tk.Toplevel(root)
    window.title(title)
    window.geometry("1000x600")
    window.configure(bg=COLORS['bg_main'])

    header = tk.Frame(window, bg=COLORS['primary'], height=70)
    header.pack(fill=tk.X)
    header.pack_propagate(False)

    tk.Label(header, text=title, font=('Segoe UI', 18, 'bold'),
             bg=COLORS['primary'], fg=COLORS['text_white']).pack(expand=True)

    main_frame = tk.Frame(window, bg=COLORS['bg_main'])
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    shadow_frame = tk.Frame(main_frame, bg=COLORS['shadow'])
    shadow_frame.pack(fill=tk.BOTH, expand=True)

    content_frame = tk.Frame(shadow_frame, bg=COLORS['bg_card'], relief=tk.FLAT, bd=0)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    text_frame = tk.Frame(content_frame, bg=COLORS['bg_card'])
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    text = tk.Text(text_frame, wrap=tk.NONE, font=('Consolas', 10),
                   bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                   relief=tk.FLAT, borderwidth=0)
    text.insert(1.0, df.round(3).to_string())
    text.config(state=tk.DISABLED)

    scrollbar_y = ttk.Scrollbar(text_frame, orient='vertical', command=text.yview)
    scrollbar_x = ttk.Scrollbar(text_frame, orient='horizontal', command=text.xview)
    text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    text.grid(row=0, column=0, sticky='nsew')
    scrollbar_y.grid(row=0, column=1, sticky='ns')
    scrollbar_x.grid(row=1, column=0, sticky='ew')

    text_frame.grid_rowconfigure(0, weight=1)
    text_frame.grid_columnconfigure(0, weight=1)

    btn_frame = tk.Frame(window, bg=COLORS['bg_main'])
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="✖ Close", font=('Segoe UI', 11, 'bold'),
             bg=COLORS['danger'], fg=COLORS['text_white'], width=15, height=1,
             relief=tk.FLAT, cursor='hand2', command=window.destroy).pack()

def show_plot_window(fig, title):
    """Display a plot in a separate modern window"""
    window = tk.Toplevel(root)
    window.title(title)
    window.geometry("1200x800")
    window.configure(bg=COLORS['bg_main'])

    header = tk.Frame(window, bg=COLORS['primary'], height=70)
    header.pack(fill=tk.X)
    header.pack_propagate(False)

    tk.Label(header, text=title, font=('Segoe UI', 18, 'bold'),
             bg=COLORS['primary'], fg=COLORS['text_white']).pack(expand=True)

    main_frame = tk.Frame(window, bg=COLORS['bg_main'])
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    shadow_frame = tk.Frame(main_frame, bg=COLORS['shadow'])
    shadow_frame.pack(fill=tk.BOTH, expand=True)

    plot_frame = tk.Frame(shadow_frame, bg=COLORS['bg_card'])
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    btn_frame = tk.Frame(window, bg=COLORS['bg_main'])
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="✖ Close", font=('Segoe UI', 11, 'bold'),
             bg=COLORS['danger'], fg=COLORS['text_white'], width=15, height=1,
             relief=tk.FLAT, cursor='hand2', command=window.destroy).pack()

def create_modern_button(parent, text, command, color, icon=""):
    """Create a modern styled button"""
    btn = tk.Button(parent, text=f"{icon} {text}" if icon else text,
                    font=('Segoe UI', 12, 'bold'), bg=color, fg=COLORS['text_white'],
                    width=30, height=2, relief=tk.FLAT, cursor='hand2',
                    activebackground=color, command=command)

    def on_enter(e):
        btn['bg'] = darken_color(color)

    def on_leave(e):
        btn['bg'] = color

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn

def darken_color(color):
    """Darken a hexadecimal color"""
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    r, g, b = max(0, r - 30), max(0, g - 30), max(0, b - 30)
    return f'#{r:02x}{g:02x}{b:02x}'

def show_frame(frame):
    """Display a frame with animation"""
    for f in [welcome_frame, menu_frame, pca_frame, ai_frame, ca_frame, security_frame]:
        f.pack_forget()
    frame.pack(fill=tk.BOTH, expand=True)

def import_excel():
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")],
        title="Select a file"
    )
    if not file_path: return
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            app_data.main_data = pd.read_excel(file_path)
        else:
            app_data.main_data = pd.read_csv(file_path)

        if "N°" in app_data.main_data.columns:
            app_data.main_data = app_data.main_data.drop(columns=["N°"])

        index_col = app_data.main_data.columns[0]
        app_data.main_data = app_data.main_data.set_index(index_col)
        app_data.main_data = clean_numeric_dataframe(app_data.main_data)

        if app_data.main_data.empty:
            messagebox.showerror("Error", "No numeric columns found")
            return

        show_dataframe_window("Data Preview", app_data.main_data)
        messagebox.showinfo("Success", f"File imported successfully!\n{len(app_data.main_data)} rows loaded")
    except Exception as e:
        messagebox.showerror("Error", f"Import error:\n{str(e)}")

def descriptive_statistics():
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import a file first")
        return

    stats_df = pd.DataFrame({
        'Mean': app_data.main_data.mean(),
        'Std_Dev': app_data.main_data.std(),
        'CV_percent': (app_data.main_data.std() / app_data.main_data.mean()) * 100
    })
    show_dataframe_window("Descriptive Statistics", stats_df)

def correlation_matrix():
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import a file first")
        return

    scaler = StandardScaler()
    app_data.scaled_data = scaler.fit_transform(app_data.main_data)
    corr = pd.DataFrame(app_data.scaled_data, columns=app_data.main_data.columns).corr()

    fig = plt.Figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111)
    sns.heatmap(corr, annot=True, cmap="RdYlBu_r", ax=ax, fmt=".2f",
                linewidths=0.5, cbar_kws={'label': 'Correlation'})
    ax.set_title("Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()

    show_plot_window(fig, "Correlation Matrix")

def scaled_matrix():
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import a file first")
        return

    scaler = StandardScaler()
    app_data.scaled_data = scaler.fit_transform(app_data.main_data)
    data_scaled_df = pd.DataFrame(app_data.scaled_data,
                              index=app_data.main_data.index,
                              columns=app_data.main_data.columns)
    show_dataframe_window("Standardized Matrix", data_scaled_df)

def calculate_inertias():
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import a file first")
        return

    scaler = StandardScaler()
    app_data.scaled_data = scaler.fit_transform(app_data.main_data)
    app_data.pca = PCA()
    app_data.pca.fit(app_data.scaled_data)

    inertias = app_data.pca.explained_variance_
    percentages = app_data.pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(percentages)

    n_comp = min(10, len(percentages))
    inertia_df = pd.DataFrame({
        'Axis': [f'F{i+1}' for i in range(n_comp)],
        'Inertia': inertias[:n_comp],
        'Percentage': percentages[:n_comp],
        'Cumulative': cumulative[:n_comp]
    }).round(4)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor='white')

    ax1.plot(range(1, len(inertias)+1), inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Axes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia', fontsize=12, fontweight='bold')
    ax1.set_title('Scree Plot', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)

    bars = ax2.bar(range(1, n_comp+1), percentages[:n_comp], color='#0288d1', alpha=0.7)
    ax2.set_xlabel('Axes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Inertia (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Inertia by Axis', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentages[i]:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    show_plot_window(fig, "Inertia Analysis")
    show_dataframe_window("Eigenvalues", inertia_df)

def factorial_plane_individuals():
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import a file first")
        return

    scaler = StandardScaler()
    app_data.scaled_data = scaler.fit_transform(app_data.main_data)
    app_data.pca = PCA()
    individuals_coords = app_data.pca.fit_transform(app_data.scaled_data)
    app_data.explained_variance = app_data.pca.explained_variance_ratio_ * 100

    app_data.individuals_coords_df = pd.DataFrame(
        individuals_coords,
        index=app_data.main_data.index.map(str),
        columns=[f"PC{i+1}" for i in range(individuals_coords.shape[1])]
    )

    fig = plt.Figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111)

    scatter = ax.scatter(app_data.individuals_coords_df["PC1"], app_data.individuals_coords_df["PC2"],
                        c=range(len(app_data.individuals_coords_df)), cmap='viridis',
                        s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

    for idx in range(len(app_data.individuals_coords_df)):
        ax.text(app_data.individuals_coords_df.iloc[idx]["PC1"],
               app_data.individuals_coords_df.iloc[idx]["PC2"],
               app_data.individuals_coords_df.index[idx], fontsize=9, fontweight='bold')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel(f"PC1 ({app_data.explained_variance[0]:.2f}%)",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f"PC2 ({app_data.explained_variance[1]:.2f}%)",
                 fontsize=12, fontweight='bold')
    ax.set_title("Factorial Plane - Individuals", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    show_plot_window(fig, "Factorial Plane - Individuals")

def correlation_circle():
    if app_data.pca is None:
        messagebox.showerror("Error", "Perform PCA first")
        return

    variables_coords = app_data.pca.components_.T
    app_data.variables_coords_df = pd.DataFrame(variables_coords[:, :2],
                                         index=app_data.main_data.columns,
                                         columns=["PC1","PC2"])

    fig = plt.Figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111)
    colors = plt.cm.Set2(range(len(app_data.variables_coords_df)))

    for idx, i in enumerate(app_data.variables_coords_df.index):
        ax.arrow(0, 0, app_data.variables_coords_df.loc[i,"PC1"],
                app_data.variables_coords_df.loc[i,"PC2"],
                head_width=0.05, head_length=0.05, fc=colors[idx],
                ec=colors[idx], linewidth=2.5, alpha=0.8)
        ax.text(app_data.variables_coords_df.loc[i,"PC1"]*1.15,
               app_data.variables_coords_df.loc[i,"PC2"]*1.15, i,
               fontsize=10, ha='center', fontweight='bold')

    circle = plt.Circle((0,0), 1, fill=False, color='#1a237e', linewidth=2, linestyle='--')
    ax.add_artist(circle)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("PC1", fontsize=12, fontweight='bold')
    ax.set_ylabel("PC2", fontsize=12, fontweight='bold')
    ax.set_title("Correlation Circle", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    show_plot_window(fig, "Correlation Circle")

def representation_quality():
    if app_data.individuals_coords_df is None:
        messagebox.showerror("Error", "Perform factorial plane first")
        return

    cos2 = (app_data.individuals_coords_df**2).div((app_data.individuals_coords_df**2).sum(axis=1), axis=0)
    cos2["Quality_PC1_PC2"] = cos2["PC1"] + cos2["PC2"]
    show_dataframe_window("Representation Quality (COS²)",
                         cos2[["PC1","PC2","Quality_PC1_PC2"]].round(3))

def contribution_individuals_variables():
    if app_data.individuals_coords_df is None or app_data.variables_coords_df is None:
        messagebox.showerror("Error", "Perform complete PCA first")
        return

    contrib_individuals = (app_data.individuals_coords_df**2).div((app_data.individuals_coords_df**2).sum(axis=0), axis=1) * 100
    contrib_variables = (app_data.variables_coords_df**2).div((app_data.variables_coords_df**2).sum(axis=0), axis=1) * 100

    show_dataframe_window("Individual Contributions (%)", contrib_individuals[["PC1","PC2"]].round(2))
    show_dataframe_window("Variable Contributions (%)", contrib_variables.round(2))

def random_forest_metrics():
    """Random Forest to predict clusters"""
    if app_data.individuals_coords_df is None:
        messagebox.showerror("Error", "Perform factorial plane (PCA) first")
        return
    if app_data.main_data is None:
        messagebox.showerror("Error", "Main data missing")
        return

    try:
        k_window = tk.Toplevel(root)
        k_window.title("Select number of clusters")
        k_window.geometry("450x350")
        k_window.configure(bg=COLORS['bg_card'])

        header = tk.Frame(k_window, bg=COLORS['primary'], height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="Choose the number of clusters",
                font=('Segoe UI', 16, 'bold'), bg=COLORS['primary'],
                fg=COLORS['text_white']).pack(expand=True)

        content = tk.Frame(k_window, bg=COLORS['bg_card'])
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        k_var = tk.IntVar(value=3)
        for k in range(3, 8):
            rb = tk.Radiobutton(content, text=f"k = {k} clusters",
                              variable=k_var, value=k,
                              font=('Segoe UI', 13), bg=COLORS['bg_card'],
                              activebackground=COLORS['bg_card'],
                              selectcolor=COLORS['secondary'])
            rb.pack(anchor='w', pady=8)

        selected_k = [None]
        def validate_k():
            selected_k[0] = k_var.get()
            k_window.destroy()

        tk.Button(content, text="✓ Validate", command=validate_k,
                 font=('Segoe UI', 13, 'bold'), bg=COLORS['success'],
                 fg=COLORS['text_white'], width=18, height=2,
                 relief=tk.FLAT, cursor='hand2').pack(pady=20)

        k_window.wait_window()

        if selected_k[0] is None:
            return
        k_clusters = selected_k[0]

        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(app_data.individuals_coords_df[['PC1', 'PC2']])

        app_data.kmeans_model = kmeans
        app_data.n_clusters = k_clusters

        feature_cols = [col for col in app_data.main_data.columns if col != 'Visitors_per_day']
        X = app_data.main_data[feature_cols].copy()
        y = clusters

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', random_state=42,
            n_jobs=-1, class_weight='balanced'
        )
        rf.fit(X_train, y_train)

        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

        app_data.rf_model = rf
        app_data.feature_cols = feature_cols
        app_data.cluster_labels = y

        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='weighted')
        precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, average='weighted')
        gap = acc_train - acc_test

        cm = confusion_matrix(y_test, y_pred_test)

        importances = pd.DataFrame({
            'Variable': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig = plt.Figure(figsize=(15, 10), facecolor='white')

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(app_data.individuals_coords_df['PC1'], app_data.individuals_coords_df['PC2'],
                   c=clusters, cmap='tab10', s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
        centers = kmeans.cluster_centers_
        ax1.scatter(centers[:, 0], centers[:, 1], c='red', s=400, marker='X',
                   linewidth=4, label='Centers', zorder=5, edgecolors='darkred')
        ax1.set_xlabel(f"PC1 ({app_data.explained_variance[0]:.1f}%)", fontweight='bold')
        ax1.set_ylabel(f"PC2 ({app_data.explained_variance[1]:.1f}%)", fontweight='bold')
        ax1.set_title(f"K-means Clustering (k={k_clusters})", fontsize=13, fontweight='bold', pad=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 2, 2)
        top_features = importances.head(10)
        colors_imp = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax2.barh(top_features['Variable'], top_features['Importance'], color=colors_imp)
        ax2.set_title("Feature Importance", fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel("Importance", fontweight='bold')

        ax3 = fig.add_subplot(2, 2, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=[f"C{i}" for i in range(k_clusters)],
                   yticklabels=[f"C{i}" for i in range(k_clusters)])
        ax3.set_title("Confusion Matrix", fontsize=13, fontweight='bold', pad=10)
        ax3.set_xlabel("Predicted", fontweight='bold')
        ax3.set_ylabel("Actual", fontweight='bold')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(range(1, 6), cv_scores, 'bo-', linewidth=2.5, markersize=10)
        ax4.axhline(cv_scores.mean(), color='red', linestyle='--',
                   label=f'Mean: {cv_scores.mean():.3f}', linewidth=2)
        ax4.fill_between(range(1, 6),
                        cv_scores.mean() - cv_scores.std(),
                        cv_scores.mean() + cv_scores.std(),
                        alpha=0.2, color='red')
        ax4.set_xlabel('Fold', fontweight='bold')
        ax4.set_ylabel('Accuracy', fontweight='bold')
        ax4.set_title("Cross-Validation (5 folds)", fontsize=13, fontweight='bold', pad=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        show_plot_window(fig, "Random Forest Results")

        cluster_names = [f"Cluster_{i}" for i in range(k_clusters)]
        report = classification_report(y_test, y_pred_test,
                                      target_names=cluster_names,
                                      output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy Train", "Accuracy Test", "CV Mean", "CV Std",
                        "Precision", "Recall", "F1-Score", "Train-Test Gap", "K-means Inertia"],
            "Value": [acc_train, acc_test, cv_scores.mean(), cv_scores.std(),
                      precision_test, recall_test, f1_test, gap, kmeans.inertia_]
        }).round(4)

        show_dataframe_window("Global Metrics", metrics_df)
        show_dataframe_window("Cluster Report", report_df.round(3))
        show_dataframe_window("Feature Importance", importances.round(4))

        success_msg = f"""RANDOM FOREST MODEL TRAINED!

Number of clusters: {k_clusters}
Test accuracy: {acc_test:.2%}
Cross-validation: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})
F1 Score: {f1_test:.2%}

Diagnostics:
• Overfitting: {'Low' if gap < 0.10 else 'Moderate' if gap < 0.15 else 'High'}
• Stability: {'Excellent' if cv_scores.std() < 0.05 else 'Good'}
        """
        messagebox.showinfo("Success", success_msg)

    except Exception as e:
        import traceback
        messagebox.showerror("Error", f"Execution error:\n{str(e)}")

def predict_new_individuals():
    """Predict cluster of new libraries via Excel import"""
    if app_data.rf_model is None:
        messagebox.showerror("Error", "Train Random Forest first!")
        return

    pred_window = tk.Toplevel(root)
    pred_window.title("Cluster Prediction")
    pred_window.geometry("1100x800")
    pred_window.configure(bg=COLORS['bg_main'])

    header = tk.Frame(pred_window, bg=COLORS['primary'], height=80)
    header.pack(fill=tk.X)
    header.pack_propagate(False)
    tk.Label(header, text="CLUSTER PREDICTION - EXCEL IMPORT",
             font=('Segoe UI', 20, 'bold'), bg=COLORS['primary'],
             fg=COLORS['text_white']).pack(expand=True)

    info_frame = tk.Frame(pred_window, bg=COLORS['bg_card'])
    info_frame.pack(fill=tk.X, padx=20, pady=15)
    info_text = f"""IMPORT an Excel file containing new libraries.
The model will predict their cluster among {app_data.n_clusters} available clusters.

REQUIRED FORMAT:
The Excel file MUST contain the following columns:
{', '.join(app_data.feature_cols)}

Optional: Add a "Library" column for names."""
    tk.Label(info_frame, text=info_text, font=('Segoe UI', 11),
             bg=COLORS['bg_card'], fg=COLORS['text_primary'],
             justify=tk.LEFT, wraplength=1000).pack(padx=15, pady=15)

    import_frame = tk.Frame(pred_window, bg=COLORS['bg_main'])
    import_frame.pack(fill=tk.X, padx=20, pady=5)

    btn_import_style = {'font': ('Segoe UI', 11, 'bold'), 'width': 18, 'height': 2,
                       'relief': tk.FLAT, 'cursor': 'hand2'}

    tk.Button(import_frame, text="IMPORT EXCEL", bg=COLORS['info'],
             fg=COLORS['text_white'], command=lambda: import_excel_file(),
             **btn_import_style).pack(side=tk.LEFT, padx=5)
    tk.Button(import_frame, text="EXPORT RESULTS", bg=COLORS['success'],
             fg=COLORS['text_white'], command=lambda: export_results(),
             **btn_import_style).pack(side=tk.LEFT, padx=5)
    tk.Button(import_frame, text="CLEAR", bg=COLORS['warning'],
             fg=COLORS['text_white'], command=lambda: clear_results(),
             **btn_import_style).pack(side=tk.LEFT, padx=5)

    results_frame = tk.Frame(pred_window, bg=COLORS['bg_main'])
    results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    canvas = tk.Canvas(results_frame, bg=COLORS['bg_main'], highlightthickness=0)
    scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_main'])

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    empty_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_card'])
    empty_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    tk.Label(empty_frame, text="Import an Excel file to see predictions",
            font=('Segoe UI', 12), bg=COLORS['bg_card'],
            fg=COLORS['text_secondary']).pack(pady=50)

    def import_excel_file():
        """Import Excel file with new libraries"""
        file_path = filedialog.askopenfilename(
            title="Select an Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            df_new = pd.read_excel(file_path)

            missing_cols = [col for col in app_data.feature_cols if col not in df_new.columns]
            if missing_cols:
                messagebox.showerror("Error",
                    f"Missing columns in Excel file:\n{', '.join(missing_cols)}")
                return

            df_features = df_new[app_data.feature_cols]

            predictions = app_data.rf_model.predict(df_features)
            probas = app_data.rf_model.predict_proba(df_features)

            pred_window.prediction_results = {
                'df_original': df_new,
                'df_features': df_features,
                'predictions': predictions,
                'probabilities': probas
            }

            display_multiple_predictions(df_new, predictions, probas, scrollable_frame)

            messagebox.showinfo("Success",
                f"{len(df_features)} library(ies) imported and predicted successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Import error:\n{str(e)}")

    def display_multiple_predictions(df_original, predictions, probas, parent_frame):
        """Display prediction results for multiple libraries"""
        for widget in parent_frame.winfo_children():
            widget.destroy()

        if len(df_original) == 0:
            return

        cluster_colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12',
                         '#9B59B6', '#1ABC9C', '#E67E22']

        for idx in range(len(df_original)):
            pred_frame = tk.Frame(parent_frame, bg=COLORS['bg_card'], relief=tk.RAISED, borderwidth=1)
            pred_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)

            header_pred = tk.Frame(pred_frame, bg=COLORS['primary'], height=30)
            header_pred.pack(fill=tk.X)
            header_pred.pack_propagate(False)

            predicted_cluster = predictions[idx]
            color = cluster_colors[predicted_cluster % len(cluster_colors)]

            library_name = f"Library #{idx+1}"

            for name_col in ['Library', 'Name', 'Label']:
                if name_col in df_original.columns:
                    library_name = str(df_original.iloc[idx][name_col])
                    break

            tk.Label(header_pred, text=f"{library_name} → Cluster {predicted_cluster}",
                    font=('Segoe UI', 11, 'bold'), bg=COLORS['primary'],
                    fg=color).pack(side=tk.LEFT, padx=10)

            content_frame = tk.Frame(pred_frame, bg=COLORS['bg_card'])
            content_frame.pack(fill=tk.X, padx=10, pady=10)

            features_text = "Features: "
            features_list = []
            for feature in app_data.feature_cols:
                features_list.append(f"{feature}: {df_original.iloc[idx][feature]}")
            features_text += " | ".join(features_list[:4])

            tk.Label(content_frame, text=features_text, font=('Segoe UI', 9),
                    bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                    wraplength=1000, justify=tk.LEFT).pack(anchor='w', pady=2)

            if len(features_list) > 4:
                features_text2 = "          " + " | ".join(features_list[4:])
                tk.Label(content_frame, text=features_text2, font=('Segoe UI', 9),
                        bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                        wraplength=1000, justify=tk.LEFT).pack(anchor='w', pady=2)

            prob_text = "Probabilities: "
            prob_list = []
            for i in range(app_data.n_clusters):
                prob_list.append(f"Cluster {i}: {probas[idx][i]:.1%}")
            prob_text += " | ".join(prob_list[:4])

            tk.Label(content_frame, text=prob_text, font=('Segoe UI', 9),
                    bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                    wraplength=1000, justify=tk.LEFT).pack(anchor='w', pady=2)

            if len(prob_list) > 4:
                prob_text2 = "               " + " | ".join(prob_list[4:])
                tk.Label(content_frame, text=prob_text2, font=('Segoe UI', 9),
                        bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                        wraplength=1000, justify=tk.LEFT).pack(anchor='w', pady=2)

    def export_results():
        """Export prediction results to Excel"""
        if not hasattr(pred_window, 'prediction_results'):
            messagebox.showwarning("No data", "No predictions to export!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Save results"
        )

        if not file_path:
            return

        try:
            results = pred_window.prediction_results
            df_export = results['df_original'].copy()

            df_export['Predicted_Cluster'] = results['predictions']

            for i in range(app_data.n_clusters):
                df_export[f'Prob_Cluster_{i}'] = results['probabilities'][:, i]

            df_export.to_excel(file_path, index=False)
            messagebox.showinfo("Success", f"Results exported to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Export error:\n{str(e)}")

    def clear_results():
        """Clear displayed results"""
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

        if hasattr(pred_window, 'prediction_results'):
            delattr(pred_window, 'prediction_results')

        empty_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_card'])
        empty_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(empty_frame, text="Import an Excel file to see predictions",
                font=('Segoe UI', 12), bg=COLORS['bg_card'],
                fg=COLORS['text_secondary']).pack(pady=50)

    close_frame = tk.Frame(pred_window, bg=COLORS['bg_main'])
    close_frame.pack(pady=15)

    tk.Button(close_frame, text="✖ CLOSE", font=('Segoe UI', 12, 'bold'),
             bg=COLORS['danger'], fg=COLORS['text_white'], width=15, height=2,
             relief=tk.FLAT, cursor='hand2',
             command=pred_window.destroy).pack()

def display_clusters_navigation():
    """Navigate clusters k=3 to k=7"""
    if app_data.individuals_coords_df is None:
        messagebox.showerror("Error", "Perform factorial plane first")
        return

    cluster_window = tk.Toplevel(root)
    cluster_window.title("Clustering Navigation")
    cluster_window.geometry("1300x850")
    cluster_window.configure(bg=COLORS['bg_main'])

    cluster_data = {'current_k': 3, 'canvas': None}

    def clear_plot():
        if cluster_data['canvas']:
            cluster_data['canvas'].get_tk_widget().destroy()

    def display_clusters(k):
        clear_plot()

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(app_data.individuals_coords_df[['PC1', 'PC2']])

        fig = plt.Figure(figsize=(13, 7), facecolor='white')
        ax = fig.add_subplot(111)

        scatter = ax.scatter(app_data.individuals_coords_df['PC1'], app_data.individuals_coords_df['PC2'],
                            c=clusters, cmap='tab10', s=180, alpha=0.7,
                            edgecolors='black', linewidth=1.5)

        for i, idx in enumerate(app_data.individuals_coords_df.index):
            ax.text(app_data.individuals_coords_df.iloc[i]['PC1'],
                   app_data.individuals_coords_df.iloc[i]['PC2'],
                   str(idx), fontsize=9, ha='center', fontweight='bold')

        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=500, marker='X',
                  linewidth=4, label='Centers', zorder=5, edgecolors='darkred')

        ax.legend(*scatter.legend_elements(), title="Clusters", loc='best', fontsize=10)
        ax.set_xlabel(f"PC1 ({app_data.explained_variance[0]:.1f}%)",
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f"PC2 ({app_data.explained_variance[1]:.1f}%)",
                     fontsize=12, fontweight='bold')
        ax.set_title(f"K-means Clustering (k = {k})",
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        cluster_data['canvas'] = FigureCanvasTkAgg(fig, master=plot_frame)
        cluster_data['canvas'].draw()
        cluster_data['canvas'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        cluster_pct = (cluster_counts / len(clusters) * 100).round(2)

        stats_text = f"STATISTICS k={k}\n" + "─" * 40 + "\n"
        for i in range(k):
            stats_text += f"Cluster {i}: {cluster_counts.get(i, 0)} points ({cluster_pct.get(i, 0)}%)\n"
        stats_text += "─" * 40 + f"\nInertia: {kmeans.inertia_:.2f}"

        stats_label.config(text=stats_text)
        k_label.config(text=f"k = {k}")
        update_nav_buttons()

    def update_nav_buttons():
        k = cluster_data['current_k']
        prev_button.config(state='normal' if k > 3 else 'disabled',
                          bg=COLORS['secondary'] if k > 3 else '#CCCCCC')
        next_button.config(state='normal' if k < 7 else 'disabled',
                          bg=COLORS['success'] if k < 7 else '#CCCCCC')

    def previous_k():
        if cluster_data['current_k'] > 3:
            cluster_data['current_k'] -= 1
            display_clusters(cluster_data['current_k'])

    def next_k():
        if cluster_data['current_k'] < 7:
            cluster_data['current_k'] += 1
            display_clusters(cluster_data['current_k'])

    header = tk.Frame(cluster_window, bg=COLORS['primary'], height=80)
    header.pack(fill=tk.X)
    header.pack_propagate(False)
    tk.Label(header, text="K-MEANS CLUSTERING NAVIGATION",
             font=('Segoe UI', 20, 'bold'), bg=COLORS['primary'],
             fg=COLORS['text_white']).pack(expand=True)

    nav_frame = tk.Frame(cluster_window, bg=COLORS['bg_main'])
    nav_frame.pack(fill=tk.X, pady=20)

    nav_buttons = tk.Frame(nav_frame, bg=COLORS['bg_main'])
    nav_buttons.pack()

    prev_button = tk.Button(nav_buttons, text="◀ Previous",
                           font=('Segoe UI', 13, 'bold'), bg=COLORS['secondary'],
                           fg=COLORS['text_white'], width=15, height=2,
                           relief=tk.FLAT, cursor='hand2', command=previous_k)
    prev_button.pack(side=tk.LEFT, padx=15)

    k_label = tk.Label(nav_buttons, text=f"k = 3",
                      font=('Segoe UI', 18, 'bold'),
                      bg=COLORS['bg_main'], fg=COLORS['accent'])
    k_label.pack(side=tk.LEFT, padx=30)

    next_button = tk.Button(nav_buttons, text="Next ▶",
                           font=('Segoe UI', 13, 'bold'), bg=COLORS['success'],
                           fg=COLORS['text_white'], width=15, height=2,
                           relief=tk.FLAT, cursor='hand2', command=next_k)
    next_button.pack(side=tk.LEFT, padx=15)

    plot_container = tk.Frame(cluster_window, bg=COLORS['bg_main'])
    plot_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    shadow_plot = tk.Frame(plot_container, bg=COLORS['shadow'])
    shadow_plot.pack(fill=tk.BOTH, expand=True)

    plot_frame = tk.Frame(shadow_plot, bg=COLORS['bg_card'])
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    stats_container = tk.Frame(cluster_window, bg=COLORS['bg_main'])
    stats_container.pack(fill=tk.X, padx=20, pady=10)

    shadow_stats = tk.Frame(stats_container, bg=COLORS['shadow'])
    shadow_stats.pack(fill=tk.X)

    stats_frame = tk.Frame(shadow_stats, bg=COLORS['bg_card'])
    stats_frame.pack(fill=tk.X, padx=2, pady=2)

    stats_label = tk.Label(stats_frame, text="", font=('Consolas', 11),
                          bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                          justify=tk.LEFT)
    stats_label.pack(padx=15, pady=15)

    tk.Button(cluster_window, text="✖ Close", font=('Segoe UI', 12, 'bold'),
             bg=COLORS['danger'], fg=COLORS['text_white'], width=15, height=1,
             relief=tk.FLAT, cursor='hand2', command=cluster_window.destroy).pack(pady=15)

    display_clusters(3)

def cluster_percentages():
    """Detailed cluster statistics"""
    if app_data.individuals_coords_df is None:
        messagebox.showerror("Error", "Perform factorial plane first")
        return

    stats_window = tk.Toplevel(root)
    stats_window.title("Cluster Statistics")
    stats_window.geometry("900x700")
    stats_window.configure(bg=COLORS['bg_main'])

    header = tk.Frame(stats_window, bg=COLORS['primary'], height=70)
    header.pack(fill=tk.X)
    header.pack_propagate(False)
    tk.Label(header, text="CLUSTER STATISTICS",
             font=('Segoe UI', 18, 'bold'), bg=COLORS['primary'],
             fg=COLORS['text_white']).pack(expand=True)

    main_frame = tk.Frame(stats_window, bg=COLORS['bg_main'])
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    canvas = tk.Canvas(main_frame, bg=COLORS['bg_main'])
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_main'])

    scrollable_frame.bind("<Configure>",
                         lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    for k in range(3, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(app_data.individuals_coords_df[['PC1', 'PC2']])

        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        cluster_pct = (cluster_counts / len(clusters) * 100).round(2)
        centers = kmeans.cluster_centers_

        shadow_frame = tk.Frame(scrollable_frame, bg=COLORS['shadow'])
        shadow_frame.pack(fill=tk.X, pady=10, padx=5)

        k_frame = tk.Frame(shadow_frame, bg=COLORS['bg_card'])
        k_frame.pack(fill=tk.X, padx=2, pady=2)

        card_header = tk.Frame(k_frame, bg=COLORS['secondary'], height=40)
        card_header.pack(fill=tk.X)
        card_header.pack_propagate(False)
        tk.Label(card_header, text=f"K = {k} CLUSTERS",
                font=('Segoe UI', 14, 'bold'), bg=COLORS['secondary'],
                fg=COLORS['text_white']).pack(expand=True)

        content = tk.Frame(k_frame, bg=COLORS['bg_card'])
        content.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(content, text=f"Total inertia: {kmeans.inertia_:.2f}",
                font=('Segoe UI', 10, 'bold'), bg=COLORS['bg_card'],
                fg=COLORS['text_primary']).pack(anchor='w', pady=5)

        for i in range(k):
            cluster_text = (f"  • Cluster {i}: {cluster_counts.get(i, 0)} libraries "
                          f"({cluster_pct.get(i, 0)}%)")
            tk.Label(content, text=cluster_text, font=('Consolas', 10),
                    bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                    anchor='w').pack(anchor='w', padx=10, pady=2)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    tk.Button(stats_window, text="✖ Close", font=('Segoe UI', 12, 'bold'),
             bg=COLORS['danger'], fg=COLORS['text_white'], width=15,
             relief=tk.FLAT, cursor='hand2', command=stats_window.destroy).pack(pady=15)

def import_contingency():
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")],
        title="Contingency Table"
    )
    if not file_path: return
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            app_data.contingency_table = pd.read_excel(file_path, index_col=0)
        else:
            app_data.contingency_table = pd.read_csv(file_path, index_col=0)

        show_dataframe_window("Contingency Table", app_data.contingency_table)
        messagebox.showinfo("Success", "Contingency table imported!")
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")

def frequency_matrix():
    if app_data.contingency_table is None:
        messagebox.showerror("Error", "Import a table first")
        return

    total = app_data.contingency_table.values.sum()
    freq_matrix = app_data.contingency_table / total
    show_dataframe_window("Frequency Matrix", freq_matrix.round(4))

def dependence_analysis():
    if app_data.contingency_table is None:
        messagebox.showerror("Error", "Import a table first")
        return

    chi2, p_value, dof, expected = chi2_contingency(app_data.contingency_table)
    n = app_data.contingency_table.sum().sum()
    inertia = chi2 / n

    result = f"""DEPENDENCE ANALYSIS (χ²)

Chi-square statistic = {chi2:.4f}
p-value = {p_value:.6f}
Degrees of freedom = {dof}
Total inertia = {inertia:.4f}

{'Significant dependence (p < 0.05)' if p_value < 0.05 else 'No dependence (p ≥ 0.05)'}
    """

    messagebox.showinfo("χ² Analysis", result)

def chi2_distance():
    if app_data.contingency_table is None:
        messagebox.showerror("Error", "Import a table first")
        return

    try:
        contingency = app_data.contingency_table.values
        n = contingency.sum()

        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)

        row_profiles = contingency / row_sums[:, np.newaxis]
        col_profiles = (contingency.T / col_sums[:, np.newaxis]).T

        row_masses = row_sums / n
        col_masses = col_sums / n

        n_rows = len(row_profiles)
        dist_rows = np.zeros((n_rows, n_rows))

        for i in range(n_rows):
            for j in range(n_rows):
                diff = row_profiles[i] - row_profiles[j]
                dist_rows[i, j] = np.sum((diff ** 2) / col_masses)

        n_cols = len(col_profiles.T)
        dist_cols = np.zeros((n_cols, n_cols))

        for i in range(n_cols):
            for j in range(n_cols):
                diff = col_profiles[:, i] - col_profiles[:, j]
                dist_cols[i, j] = np.sum((diff ** 2) / row_masses)

        dist_rows_df = pd.DataFrame(dist_rows,
                                    index=app_data.contingency_table.index,
                                    columns=app_data.contingency_table.index)

        dist_cols_df = pd.DataFrame(dist_cols,
                                    index=app_data.contingency_table.columns,
                                    columns=app_data.contingency_table.columns)

        fig = plt.Figure(figsize=(16, 7), facecolor='white')

        ax1 = fig.add_subplot(1, 2, 1)
        sns.heatmap(dist_rows_df, annot=True, fmt='.3f', cmap='YlOrRd',
                    ax=ax1, cbar_kws={'label': 'Chi-square Distance'})
        ax1.set_title('Chi-square Distances - ROW Profiles',
                     fontsize=14, fontweight='bold', pad=15)

        ax2 = fig.add_subplot(1, 2, 2)
        sns.heatmap(dist_cols_df, annot=True, fmt='.3f', cmap='YlGnBu',
                    ax=ax2, cbar_kws={'label': 'Chi-square Distance'})
        ax2.set_title('Chi-square Distances - COLUMN Profiles',
                     fontsize=14, fontweight='bold', pad=15)

        fig.tight_layout()
        show_plot_window(fig, "Chi-square Distances")

        show_dataframe_window("Row Distances", dist_rows_df)
        show_dataframe_window("Column Distances", dist_cols_df)

        max_dist_row = dist_rows_df.values[np.triu_indices_from(dist_rows_df.values, k=1)].max()

        interp = f"""CHI-SQUARE DISTANCE ANALYSIS

Maximum distance (rows): {max_dist_row:.4f}

The larger the distance, the more different the profiles.
Distance = 0 means identical profiles.

Use these distances to identify libraries
with similar or different characteristics.
        """

        messagebox.showinfo("Chi-square Distances", interp)

    except Exception as e:
        messagebox.showerror("Error", f"Error:\n{str(e)}")

def ca_factorial_plane():
    if app_data.contingency_table is None:
        messagebox.showerror("Error", "Import a table first")
        return

    try:
        contingency = app_data.contingency_table.values
        n = contingency.sum()

        correspondence_matrix = contingency / n
        row_masses = correspondence_matrix.sum(axis=1)
        col_masses = correspondence_matrix.sum(axis=0)

        expected = np.outer(row_masses, col_masses)
        residuals = (correspondence_matrix - expected) / np.sqrt(expected)

        U, S, Vt = np.linalg.svd(residuals, full_matrices=False)

        eigenvalues = S ** 2
        total_inertia = eigenvalues.sum()
        explained_variance = (eigenvalues / total_inertia) * 100
        cumulative_variance = np.cumsum(explained_variance)

        row_coords = U[:, :2] * S[:2]
        col_coords = Vt[:2, :].T * S[:2]

        fig = plt.Figure(figsize=(14, 10), facecolor='white')
        ax = fig.add_subplot(111)

        for i, label in enumerate(app_data.contingency_table.index):
            ax.scatter(row_coords[i, 0], row_coords[i, 1],
                      s=350, c='red', alpha=0.7, marker='o',
                      edgecolors='darkred', linewidth=2.5)
            ax.text(row_coords[i, 0], row_coords[i, 1],
                   f'  {label}', fontsize=10, fontweight='bold',
                   ha='left', color='darkred')

        for j, label in enumerate(app_data.contingency_table.columns):
            ax.scatter(col_coords[j, 0], col_coords[j, 1],
                      s=350, c='blue', alpha=0.7, marker='^',
                      edgecolors='darkblue', linewidth=2.5)
            ax.text(col_coords[j, 0], col_coords[j, 1],
                   f'  {label}', fontsize=9, fontweight='bold',
                   ha='left', color='darkblue')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'Axis 1 ({explained_variance[0]:.2f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Axis 2 ({explained_variance[1]:.2f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_title('CA Factorial Plane (Biplot)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(['Rows (Libraries)', 'Columns (Variables)'], fontsize=10)
        fig.tight_layout()

        show_plot_window(fig, "CA Factorial Plane")

        chi2, p_value, dof, _ = chi2_contingency(app_data.contingency_table)

        interp = f"""CORRESPONDENCE ANALYSIS

INERTIA:
   • Total: {total_inertia:.4f}
   • Axis 1: {explained_variance[0]:.2f}%
   • Axis 2: {explained_variance[1]:.2f}%
   • Plane 1-2: {cumulative_variance[1]:.2f}%

{'EXCELLENT quality' if cumulative_variance[1] >= 80 else 'GOOD quality' if cumulative_variance[1] >= 60 else 'AVERAGE quality'}
   The plane captures {cumulative_variance[1]:.1f}% of the information.

CHI-SQUARE TEST:
   Chi² = {chi2:.2f}
   p-value = {p_value:.6f}
   {'SIGNIFICANT dependence' if p_value < 0.05 else 'No dependence'}

INTERPRETATION:
   • Row/column proximity → Strong association
   • Distance from origin → Atypical profile
   • Axis direction → Dominant characteristics
        """

        messagebox.showinfo("CA", interp)

    except Exception as e:
        messagebox.showerror("Error", f"Error:\n{str(e)}")

def chi_square_test():
    if app_data.contingency_table is None:
        messagebox.showerror("Error", "Import a table first")
        return

    chi2, p_value, dof, _ = chi2_contingency(app_data.contingency_table)

    interp = f"""CHI-SQUARE TEST (χ²)

Chi-square statistic = {chi2:.4f}
p-value = {p_value:.6f}
Degrees of freedom = {dof}

INTERPRETATION:
"""
    if p_value < 0.01:
        interp += "VERY significant dependence (p < 0.01)"
    elif p_value < 0.05:
        interp += "Significant dependence (p < 0.05)"
    else:
        interp += "No dependence (p ≥ 0.05)"

    interp += "\n\nCONCLUSION:\n"
    if p_value < 0.05:
        interp += "Variables are statistically dependent.\nSignificant association detected."
    else:
        interp += "Variables are independent.\nNo significant association."

    messagebox.showinfo("Chi-square Test", interp)

def import_security_data():
    import_excel()

def isolation_forest_security():
    """Anomaly detection with Isolation Forest"""
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import data first")
        return

    try:
        security_features = app_data.main_data.copy()

        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )

        predictions = iso_forest.fit_predict(security_features)
        anomaly_scores = iso_forest.score_samples(security_features)

        anomalies = predictions == -1
        normal = predictions == 1

        results_df = pd.DataFrame({
            'Library': security_features.index,
            'Status': ['ANOMALY' if p == -1 else 'NORMAL' for p in predictions],
            'Anomaly_Score': anomaly_scores.round(3),
            'Possible_Reason': [get_security_issue_reason(row) for _, row in security_features.iterrows()]
        })

        fig = plt.Figure(figsize=(15, 8), facecolor='white')
        ax = fig.add_subplot(111)

        colors = ['red' if a else 'green' for a in anomalies]
        scatter = ax.scatter(range(len(anomaly_scores)), anomaly_scores,
                           c=colors, s=150, alpha=0.7,
                           edgecolors='black', linewidth=1.5, zorder=3)

        for i, (idx, score) in enumerate(zip(security_features.index, anomaly_scores)):
            if anomalies[i]:
                ax.text(i, score, f'  {idx}', fontsize=9, fontweight='bold',
                       ha='left', va='center', color='darkred')

        threshold = np.percentile(anomaly_scores, 10)
        ax.axhline(threshold, color='black', linestyle='--',
                  linewidth=2.5, label=f'Threshold ({threshold:.3f})')

        ax.set_xlabel('Libraries', fontsize=12, fontweight='bold')
        ax.set_ylabel('Anomaly score', fontsize=12, fontweight='bold')
        ax.set_title('Isolation Forest - Security Anomaly Detection',
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig2 = plt.Figure(figsize=(14, 10), facecolor='white')

        if anomalies.sum() > 0:
            anomaly_data = security_features[anomalies]

            median_values = security_features.median()
            deviations = ((anomaly_data - median_values) / median_values).abs().mean()

            ax2 = fig2.add_subplot(111)
            top_features = deviations.nlargest(8)
            bars = ax2.barh(range(len(top_features)), top_features.values,
                           color='#c62828', alpha=0.7)

            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features.index)
            ax2.set_xlabel('Relative deviation from median', fontsize=12, fontweight='bold')
            ax2.set_title('Variables Contributing to Anomalies',
                         fontsize=14, fontweight='bold', pad=15)

            for i, (bar, value) in enumerate(zip(bars, top_features.values)):
                ax2.text(value, bar.get_y() + bar.get_height()/2,
                        f'{value:.2%}', ha='left', va='center',
                        fontweight='bold', fontsize=10)

        fig.tight_layout()
        show_plot_window(fig, "Isolation Forest - Anomalies")

        if anomalies.sum() > 0:
            fig2.tight_layout()
            show_plot_window(fig2, "Anomaly Analysis")

        results_df_sorted = results_df.sort_values('Anomaly_Score')
        show_dataframe_window("Anomaly Detection Results", results_df_sorted)

        rapport = f"""ANOMALY DETECTION REPORT

RESULTS:
• Libraries analyzed: {len(security_features)}
• Anomalies detected: {anomalies.sum()} ({anomalies.sum()/len(security_features)*100:.1f}%)
• Average score (normal): {anomaly_scores[normal].mean():.3f}
• Average score (anomalies): {anomaly_scores[anomalies].mean():.3f}

AT-RISK LIBRARIES:
{', '.join(security_features.index[anomalies].tolist()) if anomalies.sum() > 0 else 'None'}

RECOMMENDATIONS:
1. Verify security of anomalous sites
2. Analyze activity logs
3. Control network access"""

        messagebox.showinfo("Anomaly Report", rapport)

    except Exception as e:
        messagebox.showerror("Error", f"Detection error:\n{str(e)}")

def lof_algorithm_security():
    """Anomaly detection with LOF"""
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import data first")
        return

    try:
        security_features = app_data.main_data.copy()

        n_samples = len(security_features)
        n_neighbors = min(20, max(5, n_samples // 3))

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=0.1,
            metric='euclidean',
            novelty=False
        )

        predictions = lof.fit_predict(security_features)
        lof_scores = -lof.negative_outlier_factor_

        anomalies = predictions == -1
        normal = predictions == 1

        results_df = pd.DataFrame({
            'Library': security_features.index,
            'Status': ['ANOMALY' if p == -1 else 'NORMAL' for p in predictions],
            'LOF_Score': lof_scores.round(3),
            'Risk_Level': pd.qcut(lof_scores, q=4, labels=['Low', 'Medium', 'High', 'Critical']),
            'Problematic_Features': [identify_security_issues(row) for _, row in security_features.iterrows()]
        })

        fig = plt.Figure(figsize=(15, 8), facecolor='white')
        ax = fig.add_subplot(111)

        colors = plt.cm.RdYlGn_r((lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min()))

        scatter = ax.scatter(range(len(lof_scores)), lof_scores,
                           c=lof_scores, cmap='RdYlGn_r', s=200,
                           alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)

        for i, (idx, score) in enumerate(zip(security_features.index, lof_scores)):
            if anomalies[i] or score > np.percentile(lof_scores, 75):
                ax.text(i, score, f'  {idx}', fontsize=9, fontweight='bold',
                       ha='left', va='center', color='darkred' if anomalies[i] else 'darkorange')

        ax.axhline(1.0, color='black', linestyle='--', linewidth=2,
                  label='Normal threshold (LOF=1)')
        ax.axhline(1.5, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Risk threshold (LOF=1.5)')

        ax.set_xlabel('Libraries', fontsize=12, fontweight='bold')
        ax.set_ylabel('LOF Score', fontsize=12, fontweight='bold')
        ax.set_title('LOF Algorithm - Local Anomaly Detection',
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='LOF Score')

        fig2 = plt.Figure(figsize=(14, 6), facecolor='white')

        if anomalies.sum() > 0:
            ax2 = fig2.add_subplot(121)

            normal_scores = lof_scores[normal]
            anomaly_scores = lof_scores[anomalies]

            box_data = [normal_scores, anomaly_scores]
            box = ax2.boxplot(box_data, labels=['Normal', 'Anomalies'],
                             patch_artist=True)

            box['boxes'][0].set_facecolor('lightgreen')
            box['boxes'][1].set_facecolor('lightcoral')

            ax2.set_ylabel('LOF Score', fontsize=12, fontweight='bold')
            ax2.set_title('LOF Score Comparison',
                         fontsize=13, fontweight='bold', pad=10)
            ax2.grid(True, alpha=0.3, axis='y')

            ax3 = fig2.add_subplot(122)
            risk_counts = results_df['Risk_Level'].value_counts()
            colors_risk = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
            ax3.pie(risk_counts.values, labels=risk_counts.index,
                   autopct='%1.1f%%', colors=colors_risk, startangle=90,
                   textprops={'fontweight': 'bold'})
            ax3.set_title('Risk Level Distribution',
                         fontsize=13, fontweight='bold', pad=10)

        fig.tight_layout()
        show_plot_window(fig, "LOF Algorithm - Security")

        if anomalies.sum() > 0:
            fig2.tight_layout()
            show_plot_window(fig2, "Comparative Analysis")

        results_df_sorted = results_df.sort_values('LOF_Score', ascending=False)
        show_dataframe_window("LOF Results - Risk Levels", results_df_sorted)

        rapport = f"""LOF REPORT - LOCAL ANOMALIES

DETECTION:
• Libraries analyzed: {n_samples}
• Neighbors considered: {n_neighbors}
• Anomalies detected: {anomalies.sum()} ({anomalies.sum()/n_samples*100:.1f}%)

SCORE STATISTICS:
• Global average score: {lof_scores.mean():.3f}
• Anomaly average score: {lof_scores[anomalies].mean():.3f}
• Max score: {lof_scores.max():.3f}
• Min score: {lof_scores.min():.3f}

RISK DISTRIBUTION:
• Low: {(results_df['Risk_Level'] == 'Low').sum()} sites
• Medium: {(results_df['Risk_Level'] == 'Medium').sum()} sites
• High: {(results_df['Risk_Level'] == 'High').sum()} sites
• Critical: {(results_df['Risk_Level'] == 'Critical').sum()} sites

RECOMMENDED ACTIONS:
1. Critical sites: Immediate audit
2. High sites: Security reinforcement
3. Medium sites: Increased monitoring
4. Low sites: Preventive maintenance"""

        messagebox.showinfo("LOF Report", rapport)

    except Exception as e:
        messagebox.showerror("Error", f"LOF error:\n{str(e)}")

def risk_interpretation():
    """Security risk interpretation"""
    if app_data.main_data is None:
        messagebox.showerror("Error", "Import data first")
        return

    try:
        df = app_data.main_data.copy()

        print(f"Available columns: {list(df.columns)}")

        df['Global_Score'] = 50

        if 'Server_Updates' in df.columns:
            df['Global_Score'] += df['Server_Updates'] * 0.3

        if 'Firewall_Active' in df.columns:
            df['Global_Score'] += df['Firewall_Active'] * 20

        if 'Logs_Analyzed' in df.columns:
            df['Global_Score'] += df['Logs_Analyzed'] * 0.3

        if 'Security_Incidents' in df.columns:
            df['Global_Score'] -= df['Security_Incidents'] * 5

        if 'Phishing_Attempts' in df.columns:
            df['Global_Score'] -= df['Phishing_Attempts'] * 2

        df['Global_Score'] = df['Global_Score'].clip(0, 100)

        df['Security_Level'] = pd.cut(df['Global_Score'],
                                      bins=[0, 30, 50, 70, 100],
                                      labels=['LOW', 'MEDIUM', 'GOOD', 'EXCELLENT'])

        critical_issues = []

        for idx, row in df.iterrows():
            issues = []

            if 'Firewall_Active' in df.columns and row['Firewall_Active'] == 0:
                issues.append("Firewall disabled")

            if 'Server_Updates' in df.columns and row['Server_Updates'] < 80:
                issues.append(f"Updates: {row['Server_Updates']}%")

            if 'Security_Incidents' in df.columns and row['Security_Incidents'] > 3:
                issues.append(f"Incidents: {row['Security_Incidents']}")

            if issues:
                critical_issues.append(f"{idx}: {', '.join(issues[:3])}")

        fig = plt.Figure(figsize=(14, 10), facecolor='white')

        ax1 = fig.add_subplot(2, 2, 1)
        security_counts = df['Security_Level'].value_counts()
        colors = {'EXCELLENT': 'green', 'GOOD': 'yellow',
                 'MEDIUM': 'orange', 'LOW': 'red'}
        sec_colors = [colors.get(cat, 'gray') for cat in security_counts.index]

        ax1.pie(security_counts.values, labels=security_counts.index,
               autopct='%1.1f%%', colors=sec_colors, startangle=90,
               textprops={'fontweight': 'bold'})
        ax1.set_title('Security Level Distribution',
                     fontsize=14, fontweight='bold', pad=15)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.bar(range(len(df)), df['Global_Score'].sort_values(ascending=False),
               color='#1a237e', alpha=0.7)
        ax2.set_xlabel('Libraries (sorted by score)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Security score', fontsize=12, fontweight='bold')
        ax2.set_title('Security Scores by Library',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = fig.add_subplot(2, 2, 3)
        if 'Server_Updates' in df.columns:
            ax3.barh(df.index.astype(str), df['Server_Updates'], color='#0288d1', alpha=0.7)
            ax3.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='Threshold (80%)')
            ax3.set_xlabel('% Server Updates', fontsize=12, fontweight='bold')
            ax3.set_title('Server Updates', fontsize=14, fontweight='bold', pad=15)
            ax3.legend()

        ax4 = fig.add_subplot(2, 2, 4)
        if 'Security_Incidents' in df.columns:
            incidents_data = df.nlargest(8, 'Security_Incidents')['Security_Incidents']
            ax4.barh(incidents_data.index.astype(str), incidents_data.values,
                    color='#c62828', alpha=0.7)
            ax4.set_xlabel('Number of incidents', fontsize=12, fontweight='bold')
            ax4.set_title('Top 8 - Security Incidents',
                         fontsize=14, fontweight='bold', pad=15)

        fig.tight_layout()
        show_plot_window(fig, "Security Risk Analysis")

        results_df = df[['Global_Score', 'Security_Level']].sort_values('Global_Score', ascending=False)
        show_dataframe_window("Security Scores", results_df.round(1))

        rapport = f"""SECURITY RISK ANALYSIS REPORT

GLOBAL ANALYSIS:
• Libraries analyzed: {len(df)}
• Average security score: {df['Global_Score'].mean():.1f}/100
• Dominant security level: {security_counts.index[0]}

CLASSIFICATION:
{security_counts.to_string()}

IDENTIFIED PROBLEMS ({len(critical_issues)} sites concerned):"""

        if critical_issues:
            for issue in critical_issues[:5]:
                rapport += f"\n• {issue}"
            if len(critical_issues) > 5:
                rapport += f"\n• ... and {len(critical_issues) - 5} others"
        else:
            rapport += "\n• No critical problems identified"

        rapport += f"""

RECOMMENDATIONS:

1. BASIC SECURITY:
   • Verify firewall status
   • Maintain server updates > 80%
   • Monitor security incidents

2. MONITORING:
   • Analyze logs regularly
   • Detect phishing attempts
   • Document all incidents

3. IMPROVEMENT:
   • Train staff on security
   • Update antivirus software
   • Control network access

KEY STATISTICS:"""

        if 'Server_Updates' in df.columns:
            rapport += f"\n• Average server updates: {df['Server_Updates'].mean():.1f}%"
            low_updates = (df['Server_Updates'] < 80).sum()
            rapport += f"\n• Sites with updates < 80%: {low_updates}"

        if 'Security_Incidents' in df.columns:
            rapport += f"\n• Total incidents: {df['Security_Incidents'].sum()}"
            high_incidents = (df['Security_Incidents'] > 3).sum()
            rapport += f"\n• Sites with >3 incidents: {high_incidents}"

        if 'Firewall_Active' in df.columns:
            active_firewalls = df['Firewall_Active'].sum()
            rapport += f"\n• Active firewalls: {active_firewalls}/{len(df)} ({active_firewalls/len(df)*100:.1f}%)"

        messagebox.showinfo("Risk Interpretation", rapport)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Detailed error: {error_details}")
        messagebox.showerror("Error", f"Interpretation error:\n{str(e)}\n\nDetails: {error_details[:500]}")

def get_security_issue_reason(row):
    """Identify reason for security anomaly"""
    issues = []

    if 'Firewall_Active' in row and row['Firewall_Active'] == 0:
        issues.append("No firewall")

    if 'Server_Updates' in row and row['Server_Updates'] < 70:
        issues.append(f"Low updates ({row['Server_Updates']}%)")

    if 'Logs_Analyzed' in row and row['Logs_Analyzed'] < 50:
        issues.append(f"Unanalyzed logs ({row['Logs_Analyzed']}%)")

    if 'Security_Incidents' in row and row['Security_Incidents'] > 2:
        issues.append(f"Incidents ({row['Security_Incidents']})")

    if 'Phishing_Attempts' in row and row['Phishing_Attempts'] > 3:
        issues.append(f"Phishing ({row['Phishing_Attempts']})")

    if 'Antivirus_Installed' in row and 'PC_Count' in row:
        if row['PC_Count'] > 0:
            coverage = (row['Antivirus_Installed'] / row['PC_Count']) * 100
            if coverage < 80:
                issues.append(f"Antivirus {coverage:.0f}%")

    return ", ".join(issues[:3]) if issues else "Normal configuration"

def identify_security_issues(row):
    """Identify specific security problems"""
    problems = []

    if 'Firewall_Active' in row and row['Firewall_Active'] == 0:
        problems.append("Firewall OFF")

    if 'Public_Network_Access' in row and row['Public_Network_Access'] > 20:
        problems.append("High public access")

    if 'Server_Updates' in row and row['Server_Updates'] < 60:
        problems.append("Critical updates")

    if 'Logs_Analyzed' in row and row['Logs_Analyzed'] < 40:
        problems.append("Unverified logs")

    if 'Security_Incidents' in row and row['Security_Incidents'] > 4:
        problems.append("Multi-incidents")

    if 'Phishing_Attempts' in row and row['Phishing_Attempts'] > 7:
        problems.append("Active phishing")

    return " | ".join(problems) if problems else "Security OK"

def calculate_protection_score(df):
    """Calculate protection score (0-100)"""
    score = pd.Series(0, index=df.index)
    max_score = 0

    if 'Firewall_Active' in df.columns:
        score = score.add(df['Firewall_Active'] * 25, fill_value=0)
        max_score += 25

    if 'Server_Updates' in df.columns:
        score = score.add(df['Server_Updates'] * 0.25, fill_value=0)
        max_score += 25

    if 'Antivirus_Installed' in df.columns and 'PC_Count' in df.columns:
        denominator = df['PC_Count'].replace(0, 1)
        coverage = (df['Antivirus_Installed'] / denominator) * 100
        score = score.add(coverage * 0.25, fill_value=0)
        max_score += 25

    if 'Logs_Analyzed' in df.columns:
        score = score.add(df['Logs_Analyzed'] * 0.25, fill_value=0)
        max_score += 25

    if max_score > 0:
        return (score / max_score * 100).clip(0, 100)
    return pd.Series(50, index=df.index)

def calculate_risk_score(df):
    """Calculate risk score (0-100)"""
    risk = pd.Series(0, index=df.index)
    max_risk = 0

    if 'Public_Network_Access' in df.columns:
        max_val = df['Public_Network_Access'].max()
        if max_val > 0:
            normalized = (df['Public_Network_Access'] / max_val * 25).fillna(0)
        else:
            normalized = pd.Series(0, index=df.index)
        risk = risk.add(normalized, fill_value=0)
        max_risk += 25

    if 'Phishing_Attempts' in df.columns:
        max_val = df['Phishing_Attempts'].max()
        if max_val > 0:
            normalized = (df['Phishing_Attempts'] / max_val * 25).fillna(0)
        else:
            normalized = pd.Series(0, index=df.index)
        risk = risk.add(normalized, fill_value=0)
        max_risk += 25

    if 'Security_Incidents' in df.columns:
        incidents_score = df['Security_Incidents'] * 10
        risk = risk.add(incidents_score.clip(upper=50), fill_value=0)
        max_risk += 50

    if max_risk > 0:
        return (risk / max_risk * 100).clip(0, 100)
    return pd.Series(0, index=df.index)

root = tk.Tk()
root.title("Library Data Analyst & AI")
root.geometry("1400x900")
root.configure(bg=COLORS['bg_main'])

welcome_frame = tk.Frame(root, bg=COLORS['bg_main'])

header_welcome = tk.Frame(welcome_frame, bg=COLORS['primary'], height=5)
header_welcome.pack(fill=tk.X)

content_welcome = tk.Frame(welcome_frame, bg=COLORS['bg_main'])
content_welcome.pack(expand=True)

shadow_welcome = tk.Frame(content_welcome, bg=COLORS['shadow'])
shadow_welcome.pack(padx=3, pady=3)

card_welcome = tk.Frame(shadow_welcome, bg=COLORS['bg_card'], relief=tk.FLAT, bd=0)
card_welcome.pack()

tk.Label(card_welcome, text="Library Data Analyst, AI & Cybersecurity",
         font=('Segoe UI', 32, 'bold'), bg=COLORS['bg_card'],
         fg=COLORS['primary']).pack(pady=(40, 10), padx=60)

tk.Label(card_welcome, text="LIBRARIES Project",
         font=('Segoe UI', 20), bg=COLORS['bg_card'],
         fg=COLORS['secondary']).pack(pady=(0, 40), padx=60)

start_btn = tk.Button(card_welcome, text="START",
                     font=('Segoe UI', 20, 'bold'), bg=COLORS['success'],
                     fg=COLORS['text_white'], width=18, height=2,
                     relief=tk.FLAT, cursor='hand2',
                     command=lambda: show_frame(menu_frame))
start_btn.pack(pady=(20, 40), padx=60)

footer_frame = tk.Frame(welcome_frame, bg=COLORS['bg_main'])
footer_frame.pack(side=tk.BOTTOM, pady=30)

prof_card = tk.Frame(footer_frame, bg=COLORS['bg_card'], relief=tk.FLAT, bd=2)
prof_card.pack(side=tk.LEFT, padx=20)
tk.Label(prof_card, text="Professor: Dr. EL MKHALET MOUNA",
         font=('Segoe UI', 12, 'bold'), bg=COLORS['bg_card'],
         fg=COLORS['text_primary']).pack(padx=30, pady=15)

student_card = tk.Frame(footer_frame, bg=COLORS['bg_card'], relief=tk.FLAT, bd=2)
student_card.pack(side=tk.RIGHT, padx=20)
tk.Label(student_card, text="Student Name: Fomin William",
         font=('Segoe UI', 12, 'bold'), bg=COLORS['bg_card'],
         fg=COLORS['text_primary']).pack(padx=30, pady=15)

menu_frame = tk.Frame(root, bg=COLORS['bg_main'])

header_menu = tk.Frame(menu_frame, bg=COLORS['primary'], height=90)
header_menu.pack(fill=tk.X)
header_menu.pack_propagate(False)
tk.Label(header_menu, text="MAIN MENU",
         font=('Segoe UI', 26, 'bold'), bg=COLORS['primary'],
         fg=COLORS['text_white']).pack(expand=True)

content_menu = tk.Frame(menu_frame, bg=COLORS['bg_main'])
content_menu.pack(expand=True)

pca_frame = tk.Frame(root, bg=COLORS['bg_main'])

header_pca = tk.Frame(pca_frame, bg=COLORS['secondary'], height=80)
header_pca.pack(fill=tk.X)
header_pca.pack_propagate(False)

nav_pca = tk.Frame(header_pca, bg=COLORS['secondary'])
nav_pca.pack(fill=tk.BOTH, expand=True)

tk.Button(nav_pca, text="← Back", command=lambda: show_frame(menu_frame),
         font=('Segoe UI', 11, 'bold'), bg=COLORS['primary'],
         fg=COLORS['text_white'], relief=tk.FLAT, cursor='hand2',
         width=10).pack(side=tk.LEFT, padx=20)

tk.Label(nav_pca, text="DATA-ANALYST PCA",
         font=('Segoe UI', 22, 'bold'), bg=COLORS['secondary'],
         fg=COLORS['text_white']).pack(side=tk.LEFT, padx=20)

content_pca = tk.Frame(pca_frame, bg=COLORS['bg_main'])
content_pca.pack(expand=True)

ai_frame = tk.Frame(root, bg=COLORS['bg_main'])

header_ai = tk.Frame(ai_frame, bg=COLORS['accent'], height=80)
header_ai.pack(fill=tk.X)
header_ai.pack_propagate(False)

nav_ai = tk.Frame(header_ai, bg=COLORS['accent'])
nav_ai.pack(fill=tk.BOTH, expand=True)

tk.Button(nav_ai, text="← Back", command=lambda: show_frame(menu_frame),
         font=('Segoe UI', 11, 'bold'), bg=COLORS['primary'],
         fg=COLORS['text_white'], relief=tk.FLAT, cursor='hand2',
         width=10).pack(side=tk.LEFT, padx=20)

tk.Label(nav_ai, text="ARTIFICIAL INTELLIGENCE",
         font=('Segoe UI', 22, 'bold'), bg=COLORS['accent'],
         fg=COLORS['text_white']).pack(side=tk.LEFT, padx=20)

content_ai = tk.Frame(ai_frame, bg=COLORS['bg_main'])
content_ai.pack(expand=True)

ca_frame = tk.Frame(root, bg=COLORS['bg_main'])

header_ca = tk.Frame(ca_frame, bg=COLORS['warning'], height=80)
header_ca.pack(fill=tk.X)
header_ca.pack_propagate(False)

nav_ca = tk.Frame(header_ca, bg=COLORS['warning'])
nav_ca.pack(fill=tk.BOTH, expand=True)

tk.Button(nav_ca, text="← Back", command=lambda: show_frame(menu_frame),
         font=('Segoe UI', 11, 'bold'), bg=COLORS['primary'],
         fg=COLORS['text_white'], relief=tk.FLAT, cursor='hand2',
         width=10).pack(side=tk.LEFT, padx=20)

tk.Label(nav_ca, text="DATA-ANALYST CA",
         font=('Segoe UI', 22, 'bold'), bg=COLORS['warning'],
         fg=COLORS['text_white']).pack(side=tk.LEFT, padx=20)

content_ca = tk.Frame(ca_frame, bg=COLORS['bg_main'])
content_ca.pack(expand=True)

security_frame = tk.Frame(root, bg=COLORS['bg_main'])

header_security = tk.Frame(security_frame, bg=COLORS['danger'], height=80)
header_security.pack(fill=tk.X)
header_security.pack_propagate(False)

nav_security = tk.Frame(header_security, bg=COLORS['danger'])
nav_security.pack(fill=tk.BOTH, expand=True)

tk.Button(nav_security, text="← Back", command=lambda: show_frame(menu_frame),
         font=('Segoe UI', 11, 'bold'), bg=COLORS['primary'],
         fg=COLORS['text_white'], relief=tk.FLAT, cursor='hand2',
         width=10).pack(side=tk.LEFT, padx=20)

tk.Label(nav_security, text="CYBER-SECURITY",
         font=('Segoe UI', 22, 'bold'), bg=COLORS['danger'],
         fg=COLORS['text_white']).pack(side=tk.LEFT, padx=20)

content_security = tk.Frame(security_frame, bg=COLORS['bg_main'])
content_security.pack(expand=True)

buttons_data = [
    ("DATA-ANALYST PCA", COLORS['secondary'], pca_frame),
    ("ARTIFICIAL INTELLIGENCE", COLORS['accent'], ai_frame),
    ("DATA-ANALYST CA", COLORS['warning'], ca_frame),
    ("CYBER-SECURITY", COLORS['danger'], security_frame)
]

for i, (text, color, frame_target) in enumerate(buttons_data):
    row, col = i // 2, i % 2

    shadow_btn = tk.Frame(content_menu, bg=COLORS['shadow'])
    shadow_btn.grid(row=row, column=col, padx=25, pady=25)

    btn = tk.Button(shadow_btn, text=text, font=('Segoe UI', 14, 'bold'),
                   bg=color, fg=COLORS['text_white'], width=28, height=3,
                   relief=tk.FLAT, cursor='hand2',
                   command=lambda f=frame_target: show_frame(f))
    btn.pack(padx=2, pady=2)

pca_buttons = [
    ("Import Excel", import_excel, COLORS['primary']),
    ("Descriptive Stats", descriptive_statistics, COLORS['secondary']),
    ("Standardized Matrix", scaled_matrix, COLORS['accent']),
    ("Correlation", correlation_matrix, COLORS['warning']),
    ("Inertias", calculate_inertias, COLORS['success']),
    ("Factorial Plane", factorial_plane_individuals, COLORS['danger']),
    ("Correlation Circle", correlation_circle, COLORS['primary']),
    ("Quality", representation_quality, COLORS['secondary']),
    ("Contributions", contribution_individuals_variables, COLORS['accent'])
]

for i, (text, cmd, color) in enumerate(pca_buttons):
    row, col = i // 3, i % 3

    shadow_btn = tk.Frame(content_pca, bg=COLORS['shadow'])
    shadow_btn.grid(row=row, column=col, padx=15, pady=15)

    btn = tk.Button(shadow_btn, text=text, command=cmd,
                   font=('Segoe UI', 11, 'bold'), bg=color,
                   fg=COLORS['text_white'], width=25, height=2,
                   relief=tk.FLAT, cursor='hand2')
    btn.pack(padx=2, pady=2)

ai_buttons = [
    ("Cluster Navigation (k=3→7)", display_clusters_navigation, COLORS['secondary']),
    ("Cluster Statistics", cluster_percentages, COLORS['accent']),
    ("Random Forest", random_forest_metrics, COLORS['success']),
    ("Prediction", predict_new_individuals, COLORS['danger'])
]

for i, (text, cmd, color) in enumerate(ai_buttons):
    row, col = i // 2, i % 2

    shadow_btn = tk.Frame(content_ai, bg=COLORS['shadow'])
    shadow_btn.grid(row=row, column=col, padx=25, pady=25)

    btn = tk.Button(shadow_btn, text=text, command=cmd,
                   font=('Segoe UI', 12, 'bold'), bg=color,
                   fg=COLORS['text_white'], width=35, height=2,
                   relief=tk.FLAT, cursor='hand2')
    btn.pack(padx=2, pady=2)

ca_buttons = [
    ("Import Contingency", import_contingency, COLORS['primary']),
    ("Frequencies", frequency_matrix, COLORS['secondary']),
    ("χ² Analysis", dependence_analysis, COLORS['accent']),
    ("χ² Distance", chi2_distance, COLORS['warning']),
    ("Factorial Plane", ca_factorial_plane, COLORS['success']),
    ("Chi-square Test", chi_square_test, COLORS['danger'])
]

for i, (text, cmd, color) in enumerate(ca_buttons):
    row, col = i // 2, i % 2

    shadow_btn = tk.Frame(content_ca, bg=COLORS['shadow'])
    shadow_btn.grid(row=row, column=col, padx=25, pady=25)

    btn = tk.Button(shadow_btn, text=text, command=cmd,
                   font=('Segoe UI', 12, 'bold'), bg=color,
                   fg=COLORS['text_white'], width=32, height=2,
                   relief=tk.FLAT, cursor='hand2')
    btn.pack(padx=2, pady=2)

security_buttons = [
    ("Import Data", import_security_data, COLORS['primary']),
    ("Isolation Forest", isolation_forest_security, COLORS['secondary']),
    ("LOF Algorithm", lof_algorithm_security, COLORS['accent']),
    ("Risk Interpretation", risk_interpretation, COLORS['danger'])
]

for i, (text, cmd, color) in enumerate(security_buttons):
    row, col = i // 2, i % 2

    shadow_btn = tk.Frame(content_security, bg=COLORS['shadow'])
    shadow_btn.grid(row=row, column=col, padx=25, pady=25)

    btn = tk.Button(shadow_btn, text=text, command=cmd,
                   font=('Segoe UI', 12, 'bold'), bg=color,
                   fg=COLORS['text_white'], width=32, height=2,
                   relief=tk.FLAT, cursor='hand2')
    btn.pack(padx=2, pady=2)

show_frame(welcome_frame)
root.mainloop()
