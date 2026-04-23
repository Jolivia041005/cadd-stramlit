import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import time
import re
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors, Lipinski
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
import shap
import py3Dmol
from stmol import showmol
from openai import OpenAI
from Bio import Entrez
import gc

# 页面配置
st.set_page_config(page_title="PCSK9 药物设计全流程平台", layout="wide")

DATA_DIR = "data"
PDB_PATH = os.path.join(DATA_DIR, "2P4E_prepared.pdb")
TSV_PATH = os.path.join(DATA_DIR, "BindingDB.tsv")
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 辅助函数
def get_morgan_fp(smiles, nBits=2048, radius=2):
    """生成 Morgan 指纹"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            return np.array(fp, dtype=np.float32)
    except:
        return None
    return None

def calculate_admet_properties(smiles):
    """计算 ADMET 相关理化性质"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {}
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "Lipinski_violations": sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
            Lipinski.NumHDonors(mol) > 5
        ])
    }

def calculate_docking_score(smiles):
    """模拟分子对接分数（基于 MMFF94 力场）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 999.0
        mol = Chem.AddHs(mol)
        # 嵌入 3D 构象
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return 999.0
        # 力场优化
        AllChem.MMFFOptimizeMolecule(mol)
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        if ff is None:
            return 999.0
        energy = ff.CalcEnergy()
        return round(energy, 2)  # 较低的能量表示更稳定
    except:
        return 999.0

def generate_brics_molecules(seeds, n_gen=20, min_frag_size=3):
    """基于 BRICS 片段重组生成新分子"""
    from rdkit.Chem import BRICS
    if not seeds:
        return []
    all_frags = set()
    for s in seeds:
        m = Chem.MolFromSmiles(s)
        if m:
            try:
                frags = BRICS.BRICSDecompose(m)
                all_frags.update(frags)
            except:
                continue
    frag_pool = [f for f in all_frags if Chem.MolFromSmiles(f) and Chem.MolFromSmiles(f).GetNumAtoms() >= min_frag_size]
    if len(frag_pool) < 2:
        return []
    generated = set()
    attempts = 0
    max_attempts = n_gen * 20
    while len(generated) < n_gen and attempts < max_attempts:
        attempts += 1
        f1, f2 = np.random.choice(frag_pool, 2, replace=False)
        m1, m2 = Chem.MolFromSmiles(f1), Chem.MolFromSmiles(f2)
        combined = Chem.CombineMols(m1, m2)
        smi = Chem.MolToSmiles(combined)
        if '.' in smi:
            smi = smi.replace('.', '-')
        new_mol = Chem.MolFromSmiles(smi)
        if new_mol and '.' not in Chem.MolToSmiles(new_mol):
            generated.add(Chem.MolToSmiles(new_mol))
    return list(generated)

def predict_activity(model, smiles, fp_params=None):
    """使用训练好的模型预测活性概率"""
    if fp_params is None:
        fp_params = {'nBits': 2048, 'radius': 2}
    fp = get_morgan_fp(smiles, nBits=fp_params['nBits'], radius=fp_params['radius'])
    if fp is None:
        return 0.5
    prob = model.predict_proba(fp.reshape(1, -1))[0][1]
    return prob

def display_molecule_3d(smiles, height=400, width=400, style='stick'):
    """显示分子的 3D 结构"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("无效 SMILES")
        return
    try:
        Chem.SanitizeMol(mol)
    except:
        st.warning("分子消毒失败，可能含有不合理价键，3D 结构无法生成")
        img = Draw.MolToImage(mol, size=(300,300))
        st.image(img, caption="2D structure")
        return
    mol_3d = Chem.AddHs(mol)
    try:
        # 使用默认参数，避免属性错误
        if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) != 0:
            st.warning("3D 构象嵌入失败")
            img = Draw.MolToImage(mol, size=(300,300))
            st.image(img, caption="2D structure")
            return
        AllChem.MMFFOptimizeMolecule(mol_3d)
        block = Chem.MolToMolBlock(mol_3d)
        view = py3Dmol.view(width=width, height=height)
        view.addModel(block, 'mol')
        if style == 'stick':
            view.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
        elif style == 'sphere':
            view.setStyle({'sphere': {'scale': 0.3}, 'stick': {}})
        elif style == 'line':
            view.setStyle({'line': {}})
        view.zoomTo()
        showmol(view, height=height, width=width)
    except Exception as e:
        st.warning(f"3D 显示失败: {str(e)[:100]}")
        img = Draw.MolToImage(mol, size=(300,300))
        st.image(img, caption="2D structure")

def search_pubmed(keyword, retmax=5):
    """搜索 PubMed 文献"""
    Entrez.email = "jolivia041005@gmail.com" 
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=retmax)
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_abstract(pmid):
    """获取文献摘要"""
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    records = Entrez.read(handle)
    article = records['PubmedArticle'][0]['MedlineCitation']['Article']
    title = article.get('ArticleTitle', '')
    abstract = article.get('Abstract', {}).get('AbstractText', [''])[0]
    return title, abstract

def plot_learning_curve(estimator, X, y, cv=5, scoring='roc_auc'):
    """绘制学习曲线"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig, ax = plt.subplots()
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score (ROC-AUC)")
    ax.legend(loc="best")
    return fig

# 内置药物库 (中文名称)
def get_drug_library():
    """返回扩展的药物库（包含常见他汀类、贝特类、PCSK9相关分子等）"""
    return pd.DataFrame([
        {"name": "阿托伐他汀 (Atorvastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3"},
        {"name": "瑞舒伐他汀 (Rosuvastatin)", "smiles": "COC1=C(C=C(C=C1)C(C)C(=O)O)C2=CC=C(C=C2)F"},
        {"name": "辛伐他汀 (Simvastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3"},
        {"name": "普伐他汀 (Pravastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3O"},
        {"name": "氟伐他汀 (Fluvastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3F"},
        {"name": "匹伐他汀 (Pitavastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3C#N"},
        {"name": "依折麦布 (Ezetimibe)", "smiles": "FC1=CC=C(C=C1)[C@@H]2[C@H](C3=CC=C(C=C3)N2C(=O)C4=CC=CC=C4)O"},
        {"name": "非诺贝特 (Fenofibrate)", "smiles": "CC(C)OC(=O)C(C)(C)OC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)Cl"},
        {"name": "吉非罗齐 (Gemfibrozil)", "smiles": "CC(C)CC1=CC=C(C=C1)OCC(C)(C)C(=O)O"},
        {"name": "苯扎贝特 (Bezafibrate)", "smiles": "CC(C)(C)C(=O)OC1=CC=C(C=C1)C(=O)NC(CC2=CC=CC=C2)C(=O)O"},
        {"name": "氯贝丁酯 (Clofibrate)", "smiles": "CC(C)(C)C(=O)OC1=CC=C(C=C1)C(=O)OC"},
        {"name": "阿利西尤单抗模拟小分子", "smiles": "CC(C)(C)C1=CC=C(C=C1)C2=CC=CC=C2C(=O)NCCN3CCOCC3"},
        {"name": "依洛尤单抗模拟小分子", "smiles": "C1CCN(CC1)C2=CC=C(C=C2)C3=CC=CC=C3C(=O)NCCN4CCOCC4"},
        {"name": "PCSK9 抑制剂模拟 (PF-06446846)", "smiles": "CC(C)(C)NC(=O)C1=CC=C(C=C1)NC(=O)C2=CC=CC=C2C3=CC=CC=C3"},
        {"name": "小分子抑制剂模拟 (SBC-115076)", "smiles": "CC(C)(C)C1=CC=C(C=C1)C2=CC=CC=C2C(=O)NC3=CC=CC=C3C(=O)O"},
        {"name": "PCSK9 配体模拟 (化合物 1)", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C3=CC=CC=C3"},
        {"name": "PCSK9 配体模拟 (化合物 2)", "smiles": "CCOC(=O)C1=CC=C(C=C1)C2=CC=CC=C2C(=O)NCCN3CCOCC3"},
    ])

# 主程序
def main():
    st.sidebar.title("PCSK9 药物设计平台")
    st.sidebar.markdown("---")
    
    # 全局状态初始化
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'generated_mols' not in st.session_state:
        st.session_state.generated_mols = []
    if 'active_seeds' not in st.session_state:
        st.session_state.active_seeds = []
    if 'fp_params' not in st.session_state:
        st.session_state.fp_params = {'nBits': 2048, 'radius': 2}
    if 'ic50_threshold' not in st.session_state:
        st.session_state.ic50_threshold = 1000  # 默认 1000 nM

    # 文件读取模块
    st.sidebar.subheader("数据加载")
    if st.sidebar.button("读取 BindingDB 和 PDB 文件"):
        if os.path.exists(TSV_PATH) and os.path.exists(PDB_PATH):
            try:
                df = pd.read_csv(TSV_PATH, sep='\t', usecols=['Ligand SMILES', 'IC50 (nM)'], on_bad_lines='skip')
                df.columns = ['smiles', 'ic50']
                df['ic50'] = pd.to_numeric(df['ic50'].str.extract(r'(\d+)')[0], errors='coerce')
                df = df.dropna().drop_duplicates('smiles')
                # 使用当前阈值标记标签
                df['label'] = (df['ic50'] <= st.session_state.ic50_threshold).astype(int)
                st.session_state.raw_df = df
                st.session_state.data_loaded = True
                st.success(f"数据加载成功！共 {len(df)} 个分子，活性分子 {df['label'].sum()} 个（IC50 ≤ {st.session_state.ic50_threshold} nM）。")
                st.session_state.active_seeds = df[df['label']==1]['smiles'].tolist()
            except Exception as e:
                st.error(f"数据加载失败: {e}")
        else:
            st.error(f"文件缺失：请确保 {TSV_PATH} 和 {PDB_PATH} 存在")

    # 侧边栏导航
    if not st.session_state.data_loaded:
        st.info("请先点击左侧「读取 BindingDB 和 PDB 文件」开始")
        st.stop()

    menu = st.sidebar.radio(
        "功能导航",
        ["靶点识别", "数据探索与QSAR建模", "分子设计", "虚拟筛选(ADMET+对接)",
         "药物重定位", "耐药性分析", "知识提取", "分子3D展示"]
    )

    # 1. 靶点识别 
    if menu == "靶点识别":
        st.header("1. PCSK9 靶点结构分析 (PDB: 2P4E)")
        if os.path.exists(PDB_PATH):
            with open(PDB_PATH, "r") as f:
                pdb_data = f.read()
            view = py3Dmol.view(width=800, height=500)
            view.addModel(pdb_data, 'pdb')
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.addStyle({'resi': '461'}, {'stick': {'colorscheme': 'redCarbon'}})
            view.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'lightblue'})
            view.zoomTo()
            showmol(view, height=500, width=800)
            st.info("红色棒状: TRP461 关键残基；蓝色半透明表面: 分子表面")
            with st.expander("查看关键残基信息"):
                st.markdown("""
                - **TRP461**: 位于活性口袋中心，与配体形成 π-π 堆积作用。
                - **ARG458**: 与配体形成氢键，稳定结合构象。
                - **LEU436**: 疏水相互作用的重要残基。
                """)
        else:
            st.error("PDB 文件未找到，请检查 data/ 目录")

    # 2. 数据探索与QSAR建模
    elif menu == "数据探索与QSAR建模":
        st.header("2. 数据探索与可交互 QSAR 建模")
        df = st.session_state.raw_df
        
        # 可交互的 IC50 阈值滑块
        st.subheader("活性阈值设置")
        new_threshold = st.slider("IC50 阈值 (nM)", min_value=10, max_value=10000, value=st.session_state.ic50_threshold, step=50, 
                                   help="小于等于该值的分子视为活性正样本")
        if new_threshold != st.session_state.ic50_threshold:
            st.session_state.ic50_threshold = new_threshold
            # 重新标记标签
            df['label'] = (df['ic50'] <= new_threshold).astype(int)
            st.session_state.raw_df = df
            st.session_state.active_seeds = df[df['label']==1]['smiles'].tolist()
            st.rerun()
        
        # 数据概览
        st.subheader("当前数据概览")
        col1, col2, col3 = st.columns(3)
        col1.metric("总分子数", len(df))
        col2.metric(f"活性分子 (IC50 ≤ {st.session_state.ic50_threshold} nM)", df['label'].sum())
        col3.metric("非活性分子", len(df) - df['label'].sum())
        
        # 活性分布图 (使用英文标签)
        fig, ax = plt.subplots()
        sns.histplot(df['ic50'], bins=50, log_scale=True, ax=ax)
        ax.axvline(x=st.session_state.ic50_threshold, color='r', linestyle='--', label=f'Threshold = {st.session_state.ic50_threshold} nM')
        ax.set_xlabel("IC50 (nM, log scale)")
        ax.set_title("IC50 Distribution")
        ax.legend()
        st.pyplot(fig)
        
        # 交互式参数设置
        st.subheader("QSAR 模型参数设置")
        with st.expander("点击展开参数调节面板"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                n_estimators = st.slider("随机森林树数量", 50, 300, 100, step=50)
                max_depth = st.slider("最大深度", 1, 20, 3)
            with col_b:
                fp_nbits = st.slider("指纹位数", 512, 4096, 2048, step=512)
                fp_radius = st.slider("指纹半径", 1, 4, 2)
            with col_c:
                test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, step=0.05)
                class_weight = st.selectbox("类别权重", ["balanced", "balanced_subsample", None])
            use_grid_search = st.checkbox("使用网格搜索优化超参数（耗时较长）")
        
        # 负样本平衡比例
        ratio = st.slider("正负样本比例 (1 : N)", 1, 10, 5, help="例如 1:5 表示负样本是正样本的 5 倍")
        effective_ratio = min(ratio, 5)
        
        if st.button("开始训练模型"):
            pos = df[df['label']==1]['smiles'].tolist()
            neg_pool = df[df['label']==0]['smiles'].tolist()
            target_neg = len(pos) * effective_ratio
            with st.spinner(f"正在平衡数据 (1:{effective_ratio}) 并提取指纹..."):
                if len(neg_pool) < target_neg:
                    neg_samples = list(np.random.choice(neg_pool, target_neg, replace=True))
                else:
                    neg_samples = list(np.random.choice(neg_pool, target_neg, replace=False))
                all_smiles = pos + neg_samples
                all_labels = [1]*len(pos) + [0]*len(neg_samples)
                # 提取指纹
                valid_fps = []
                valid_labels = []
                progress_bar = st.progress(0)
                for i, (smi, lab) in enumerate(zip(all_smiles, all_labels)):
                    fp = get_morgan_fp(smi, nBits=fp_nbits, radius=fp_radius)
                    if fp is not None:
                        valid_fps.append(fp)
                        valid_labels.append(lab)
                    progress_bar.progress((i+1)/len(all_smiles))
                X = np.array(valid_fps, dtype=np.float32)
                y = np.array(valid_labels)
                st.write(f"有效指纹数: {len(X)} / {len(all_smiles)}")
                
                # 清理中间变量
                del valid_fps, all_smiles, all_labels, pos, neg_samples
                gc.collect()
                
                # 划分训练测试集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
                
                # 训练模型
                if use_grid_search:
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20],
                        'min_samples_split': [2, 5]
                    }
                    rf = RandomForestClassifier(class_weight=class_weight, random_state=42, n_jobs=1)
                    grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
                    grid.fit(X_train, y_train)
                    best_rf = grid.best_estimator_
                    st.write(f"最佳参数: {grid.best_params_}")
                else:
                    best_rf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight=class_weight,
                        random_state=42,
                        n_jobs=1
                    )
                    best_rf.fit(X_train, y_train)
                
                # 评估
                y_pred = best_rf.predict(X_test)
                y_proba = best_rf.predict_proba(X_test)[:,1]
                auc = roc_auc_score(y_test, y_proba)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"模型训练完成！AUC = {auc:.4f}, Accuracy = {acc:.4f}")
                st.session_state.model = best_rf
                st.session_state.fp_params = {'nBits': fp_nbits, 'radius': fp_radius}
                joblib.dump(best_rf, os.path.join(MODEL_DIR, "qsar_model.pkl"))
                
                # 绘制 ROC 曲线和 PR 曲线 (英文标签)
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
                ax1.plot(fpr, tpr, label=f'RF (AUC={auc:.3f})')
                ax1.plot([0,1], [0,1], 'k--')
                ax1.set_xlabel('False Positive Rate')
                ax1.set_ylabel('True Positive Rate')
                ax1.set_title('ROC Curve')
                ax1.legend()
                ax2.plot(recall, precision, label=f'PR (AUC={auc:.3f})')
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
                ax2.set_title('Precision-Recall Curve')
                ax2.legend()
                st.pyplot(fig1)
                
                # 混淆矩阵 (英文标签)
                cm = confusion_matrix(y_test, y_pred)
                fig2, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                st.pyplot(fig2)
                
                # 特征重要性（前30）
                importances = best_rf.feature_importances_
                indices = np.argsort(importances)[-30:]
                fig3, ax = plt.subplots(figsize=(10,6))
                ax.barh(range(len(indices)), importances[indices])
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([f"FP_{i}" for i in indices])
                ax.set_xlabel('Feature Importance')
                st.pyplot(fig3)
                
                # 学习曲线
                st.subheader("学习曲线")
                fig_lc = plot_learning_curve(best_rf, X, y, cv=3)
                st.pyplot(fig_lc)
                
                # 释放大型数组
                del X, y, X_train, X_test, y_train, y_test
                gc.collect()
                
                # t-SNE 降维可视化（可选）
                if st.checkbox("显示 t-SNE 降维图（可能需要几十秒，内存消耗较大）"):
                    with st.spinner("计算 t-SNE..."):
                        # 对原始指纹进行采样，如果样本太多则只取前5000个
                        if len(X) > 5000:
                            st.warning("样本数超过5000，将随机采样5000个进行 t-SNE")
                            sample_idx = np.random.choice(len(X), 5000, replace=False)
                            X_tsne_input = X[sample_idx]
                            y_tsne = y[sample_idx]
                        else:
                            X_tsne_input = X
                            y_tsne = y
                        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                        X_tsne = tsne.fit_transform(X_tsne_input)
                        fig4, ax = plt.subplots()
                        scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_tsne, cmap='coolwarm', alpha=0.6)
                        ax.set_title('t-SNE visualization of fingerprints')
                        plt.colorbar(scatter, label='Activity')
                        st.pyplot(fig4)
                        del X_tsne_input, X_tsne, y_tsne
                        gc.collect()

    # 3. 分子设计
    elif menu == "分子设计":
        st.header("3. 基于 BRICS 的分子生成")
        col1, col2 = st.columns(2)
        with col1:
            n_gen = st.slider("生成分子数量", 10, 100, 30, step=5)
            min_frag_size = st.slider("最小片段原子数", 2, 8, 3)
        with col2:
            use_actives = st.multiselect("选择种子分子（活性分子）", st.session_state.active_seeds, default=st.session_state.active_seeds[:10])
        if st.button("开始生成"):
            if not use_actives:
                st.warning("请至少选择一个种子分子")
            else:
                with st.spinner("正在进行片段重组..."):
                    new_mols = generate_brics_molecules(use_actives, n_gen=n_gen, min_frag_size=min_frag_size)
                    st.session_state.generated_mols = new_mols
                    st.success(f"生成了 {len(new_mols)} 个新分子")
                    # 显示分子网格
                    mols = [Chem.MolFromSmiles(s) for s in new_mols if Chem.MolFromSmiles(s)]
                    if mols:
                        cols_per_row = st.slider("每行分子数", 2, 6, 4)
                        img = Draw.MolsToGridImage(mols[:24], molsPerRow=cols_per_row, subImgSize=(200,200))
                        st.image(img)
                    else:
                        st.warning("无有效分子生成，请尝试调整参数或种子分子")
        
        # 显示已生成的分子列表
        if st.session_state.generated_mols:
            st.subheader("已生成的分子库")
            st.dataframe(pd.DataFrame({"SMILES": st.session_state.generated_mols}))

    # 4. 虚拟筛选 
    elif menu == "虚拟筛选(ADMET+对接)":
        st.header("4. 虚拟筛选：QSAR + ADMET + 对接分数")
        if not st.session_state.generated_mols:
            st.warning("请先在「分子设计」中生成分子")
        else:
            if st.session_state.model is None:
                st.warning("请先在「数据探索与QSAR建模」中训练模型")
            else:
                # 筛选参数
                st.subheader("筛选条件设置")
                col1, col2 = st.columns(2)
                with col1:
                    min_act_prob = st.slider("最小活性概率", 0.0, 1.0, 0.5, 0.05)
                    max_lipinski_viol = st.slider("最大 Lipinski 违例数", 0, 4, 1)
                with col2:
                    weight_qsar = st.slider("QSAR 概率权重", 0.0, 1.0, 0.7, 0.05)
                    weight_docking = 1.0 - weight_qsar
                
                if st.button("开始虚拟筛选"):
                    results = []
                    progress_bar = st.progress(0)
                    for idx, smi in enumerate(st.session_state.generated_mols):
                        prop = calculate_admet_properties(smi)
                        if prop.get("Lipinski_violations", 0) > max_lipinski_viol:
                            continue
                        act_prob = predict_activity(st.session_state.model, smi, st.session_state.fp_params)
                        if act_prob < min_act_prob:
                            continue
                        docking = calculate_docking_score(smi)
                        # 对接分数归一化（使用 sigmoid，较高能量对应低分）
                        docking_norm = 1 / (1 + np.exp(docking / 5))  # 能量高时得分接近0，能量低时接近1
                        combined_score = weight_qsar * act_prob + weight_docking * docking_norm
                        results.append({
                            "SMILES": smi,
                            "活性概率": round(act_prob, 4),
                            "对接分数(kcal/mol)": docking,
                            "综合得分": round(combined_score, 4),
                            "MW": prop.get("MW"),
                            "LogP": prop.get("LogP"),
                            "HBD": prop.get("HBD"),
                            "HBA": prop.get("HBA"),
                            "TPSA": prop.get("TPSA"),
                            "Lipinski违例": prop.get("Lipinski_violations")
                        })
                        progress_bar.progress((idx+1)/len(st.session_state.generated_mols))
                    
                    df_res = pd.DataFrame(results).sort_values("综合得分", ascending=False)
                    st.dataframe(df_res)
                    # 交互性增强：让用户选择要显示3D的分子
                    if not df_res.empty:
                        st.subheader("分子3D展示（可交互）")
                        # 创建下拉选择框，显示 SMILES 和综合得分
                        selected_smi = st.selectbox(
                            "选择要查看3D结构的分子",
                            options=df_res["SMILES"].tolist(),
                            format_func=lambda x: f"{x[:50]}... (Score: {df_res[df_res['SMILES']==x]['综合得分'].values[0]})"
                        )
                        if selected_smi:
                            display_molecule_3d(selected_smi, height=400, width=400)
                            # 显示该分子的对接分数
                            docking_val = df_res[df_res['SMILES']==selected_smi]['对接分数(kcal/mol)'].values[0]
                            st.caption(f"对接分数: {docking_val} kcal/mol (越低越稳定)")

    # 5. 药物重定位
    elif menu == "药物重定位":
        st.header("5. 已上市药物对 PCSK9 的活性预测")
        if st.session_state.model is None:
            st.warning("请先训练 QSAR 模型")
        else:
            drug_library = get_drug_library()
            st.dataframe(drug_library)
            if st.button("预测重定位潜力"):
                results = []
                for _, row in drug_library.iterrows():
                    prob = predict_activity(st.session_state.model, row["smiles"], st.session_state.fp_params)
                    results.append({"药物名称": row["name"], "预测活性概率": round(prob, 4)})
                df_drug = pd.DataFrame(results).sort_values("预测活性概率", ascending=False)
                st.dataframe(df_drug)
                # 绘制条形图 (英文标签)
                fig, ax = plt.subplots(figsize=(8,6))
                sns.barplot(data=df_drug, x="预测活性概率", y="药物名称", ax=ax)
                ax.set_xlim(0,1)
                ax.set_xlabel("Predicted PCSK9 Activity Probability")
                ax.set_title("Drug Repurposing Potential Prediction")
                st.pyplot(fig)

    # 6. 耐药性分析
    elif menu == "耐药性分析":
        st.header("6. 分子结构扰动下的活性稳定性评估")
        if not st.session_state.generated_mols:
            st.warning("请先生成分子")
        else:
            target_smi = st.selectbox("选择待分析分子", st.session_state.generated_mols)
            if target_smi and st.session_state.model:
                # 扰动参数
                noise_level = st.slider("指纹位点突变概率", 0.01, 0.2, 0.05, 0.01)
                n_trials = st.slider("扰动次数", 10, 100, 50)
                fp = get_morgan_fp(target_smi, nBits=st.session_state.fp_params['nBits'], radius=st.session_state.fp_params['radius'])
                if fp is not None:
                    original_prob = st.session_state.model.predict_proba(fp.reshape(1, -1))[0][1]
                    st.metric("原始活性概率", f"{original_prob:.4f}")
                    drift_probs = []
                    for _ in range(n_trials):
                        noise = np.random.binomial(1, noise_level, fp.shape)
                        noisy_fp = np.clip(fp + noise, 0, 1)
                        new_prob = st.session_state.model.predict_proba(noisy_fp.reshape(1, -1))[0][1]
                        drift_probs.append(new_prob)
                    fig, ax = plt.subplots()
                    ax.plot(drift_probs, marker='o', linestyle='-', alpha=0.7)
                    ax.axhline(y=original_prob, color='r', linestyle='--', label='Original probability')
                    ax.set_xlabel("Perturbation index")
                    ax.set_ylabel("Predicted activity probability")
                    ax.set_title(f"Resistance stress test ({noise_level*100}% fingerprint mutation)")
                    ax.legend()
                    st.pyplot(fig)
                    drift_std = np.std(drift_probs)
                    st.write(f"Activity probability std: {drift_std:.4f}")
                    if drift_std < 0.05:
                        st.success("Low drift, good resistance potential")
                    else:
                        st.warning("High drift, prone to drug resistance")

    # 7. 知识提取
    elif menu == "知识提取":
        st.header("7. 文献知识提取 (OpenAI + PubMed)")
        openai_key = st.text_input("请输入 OpenAI API Key", type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            client = OpenAI()
            keyword = st.text_input("输入关键词 (例如: PCSK9 inhibitor toxicity)", value="PCSK9 inhibitor adverse effects")
            if st.button("检索文献并提取"):
                with st.spinner("正在检索 PubMed..."):
                    pmids = search_pubmed(keyword, retmax=3)
                    if not pmids:
                        st.warning("未找到相关文献")
                    else:
                        for pmid in pmids:
                            title, abstract = fetch_abstract(pmid)
                            st.markdown(f"**PMID: {pmid}**")
                            st.markdown(f"*{title}*")
                            st.caption(abstract[:500] + "...")
                            # 提取毒性信息
                            prompt = f"""从以下文献摘要中提取与化合物毒副作用相关的信息，按照 TSV 格式输出：
化合物\t类型\t毒副作用
如果无相关信息则留空。
摘要：{abstract}"""
                            try:
                                response = client.responses.create(
                                    model="gpt-3.5-turbo",
                                    input=prompt
                                )
                                extracted = response.output_text
                                st.text_area("提取结果", extracted, height=150)
                                try:
                                    lines = extracted.strip().split('\n')
                                    if len(lines) > 1:
                                        data = "\n".join(lines[1:])  # 跳过表头
                                        df = pd.read_csv(StringIO(data), sep='\t', names=["化合物","类型","毒副作用"])
                                        st.dataframe(df)
                                except:
                                    pass
                            except Exception as e:
                                st.error(f"OpenAI 调用失败: {e}")
                            st.divider()
        else:
            st.info("请输入 OpenAI API Key 以启用文献知识提取")

    # 分子3D展示
    elif menu == "分子3D展示":
        st.header("8. 分子 3D 构象展示")
        option = st.radio("选择分子来源", ["从生成库中选择", "手动输入 SMILES", "从活性种子中选择", "从药物库中选择"])
        smiles_input = ""
        if option == "从生成库中选择":
            if st.session_state.generated_mols:
                smiles_input = st.selectbox("选择分子", st.session_state.generated_mols)
            else:
                st.warning("生成库为空，请先运行分子设计")
                smiles_input = ""
        elif option == "从活性种子中选择":
            if st.session_state.active_seeds:
                smiles_input = st.selectbox("选择活性分子", st.session_state.active_seeds)
            else:
                st.warning("无活性种子")
        elif option == "从药物库中选择":
            drug_library = get_drug_library()
            drug_name = st.selectbox("选择药物", drug_library["name"].tolist())
            smiles_input = drug_library[drug_library["name"] == drug_name]["smiles"].values[0]
        else:
            smiles_input = st.text_input("输入 SMILES", "CC(=O)NC1=CC=CC=C1")
        
        if smiles_input:
            style = st.selectbox("3D 显示样式", ["stick", "sphere", "line"])
            display_molecule_3d(smiles_input, height=500, width=500, style=style)
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                img = Draw.MolToImage(mol, size=(300,300))
                st.image(img, caption="2D 结构")
                props = calculate_admet_properties(smiles_input)
                if props:
                    st.write("**理化性质**")
                    st.json(props)

if __name__ == "__main__":
    main()
