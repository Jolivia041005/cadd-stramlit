import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import time
import re
import gc  # 显式内存回收
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
            # 使用 float32 节省内存
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
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return 999.0
        AllChem.MMFFOptimizeMolecule(mol)
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        if ff is None:
            return 999.0
        energy = ff.CalcEnergy()
        return round(energy, 2)
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
    mol_3d = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) == 0:
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
    else:
        st.warning("3D 构象生成失败")

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
    abstract_text = article.get('Abstract', {}).get('AbstractText', [''])
    abstract = abstract_text[0] if abstract_text else ""
    return title, abstract

def plot_learning_curve(estimator, X, y, cv=5, scoring='roc_auc'):
    """绘制学习曲线"""
    # 限制并行数以节省内存
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42)
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

# 内置药物库
def get_drug_library():
    """返回扩展的药物库"""
    return pd.DataFrame([
        {"name": "阿托伐他汀 (Atorvastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3"},
        {"name": "瑞舒伐他汀 (Rosuvastatin)", "smiles": "COC1=C(C=C(C=C1)C(C)C(=O)O)C2=CC=C(C=C2)F"},
        {"name": "辛伐他汀 (Simvastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3"},
        {"name": "普伐他汀 (Pravastatin)", "smiles": "CC(C)(C)C(C(=O)O)CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3O"},
        {"name": "依折麦布 (Ezetimibe)", "smiles": "FC1=CC=C(C=C1)[C@@H]2[C@H](C3=CC=C(C=C3)N2C(=O)C4=CC=CC=C4)O"},
        {"name": "非诺贝特 (Fenofibrate)", "smiles": "CC(C)OC(=O)C(C)(C)OC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)Cl"},
        {"name": "PCSK9 抑制剂模拟 (PF-06446846)", "smiles": "CC(C)(C)NC(=O)C1=CC=C(C=C1)NC(=O)C2=CC=CC=C2C3=CC=CC=C3"},
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
        st.session_state.ic50_threshold = 1000

    # 文件读取模块
    st.sidebar.subheader("数据加载")
    use_example = st.sidebar.checkbox("使用示例数据", value=False)
    if st.sidebar.button("读取 BindingDB 和 PDB 文件"):
        if use_example:
            np.random.seed(42)
            n_samples = 400
            smiles_list = ["CC(=O)NC1=CC=CC=C1", "C1=CC=CC=C1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"] * (n_samples // 3 + 1)
            smiles_list = smiles_list[:n_samples]
            ic50 = np.random.uniform(10, 10000, n_samples)
            df = pd.DataFrame({"smiles": smiles_list, "ic50": ic50})
            df['label'] = (df['ic50'] <= st.session_state.ic50_threshold).astype(int)
            st.session_state.raw_df = df
            st.session_state.data_loaded = True
            st.session_state.active_seeds = df[df['label']==1]['smiles'].tolist()
            st.success("示例数据已生成")
        elif os.path.exists(TSV_PATH) and os.path.exists(PDB_PATH):
            try:
                df = pd.read_csv(TSV_PATH, sep='\t', usecols=['Ligand SMILES', 'IC50 (nM)'], on_bad_lines='skip')
                df.columns = ['smiles', 'ic50']
                df['ic50'] = pd.to_numeric(df['ic50'].str.extract(r'(\d+)')[0], errors='coerce')
                df = df.dropna().drop_duplicates('smiles')
                df['label'] = (df['ic50'] <= st.session_state.ic50_threshold).astype(int)
                st.session_state.raw_df = df
                st.session_state.data_loaded = True
                st.session_state.active_seeds = df[df['label']==1]['smiles'].tolist()
                st.success("数据加载成功")
            except Exception as e:
                st.error(f"加载失败: {e}")
        else:
            st.error("找不到文件")

    if not st.session_state.data_loaded:
        st.info("请先加载数据")
        st.stop()

    menu = st.sidebar.radio("功能导航", ["靶点识别", "数据探索与QSAR建模", "分子设计", "虚拟筛选(ADMET+对接)", "药物重定位", "耐药性分析", "知识提取", "分子3D展示"])

    if menu == "靶点识别":
        st.header("1. PCSK9 靶点结构分析")
        if os.path.exists(PDB_PATH):
            with open(PDB_PATH, "r") as f:
                pdb_data = f.read()
            view = py3Dmol.view(width=800, height=500)
            view.addModel(pdb_data, 'pdb')
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.addStyle({'resi': '461'}, {'stick': {'colorscheme': 'redCarbon'}})
            view.zoomTo()
            showmol(view, height=500, width=800)
        else:
            st.error("PDB缺失")

    elif menu == "数据探索与QSAR建模":
        st.header("2. QSAR 建模")
        df = st.session_state.raw_df
        new_threshold = st.slider("IC50 阈值", 10, 10000, st.session_state.ic50_threshold)
        if new_threshold != st.session_state.ic50_threshold:
            st.session_state.ic50_threshold = new_threshold
            df['label'] = (df['ic50'] <= new_threshold).astype(int)
            st.rerun()
            
        ratio = st.slider("负样本采样比例 (1 : N)", 1, 10, 5, help="限制负样本数量以保护内存")
        
        if st.button("开始训练模型"):
            pos = df[df['label']==1]['smiles'].tolist()
            neg_pool = df[df['label']==0]['smiles'].tolist()
            
            target_neg = min(len(neg_pool), len(pos) * ratio)
            neg_samples = list(np.random.choice(neg_pool, target_neg, replace=False))
            
            all_smiles = pos + neg_samples
            all_labels = [1]*len(pos) + [0]*len(neg_samples)
            
            with st.spinner("提取特征中..."):
                valid_fps = []
                valid_labels = []
                for smi, lab in zip(all_smiles, all_labels):
                    fp = get_morgan_fp(smi, nBits=st.session_state.fp_params['nBits'])
                    if fp is not None:
                        valid_fps.append(fp)
                        valid_labels.append(lab)
                
                X = np.array(valid_fps, dtype=np.float32)
                y = np.array(valid_labels)
                del valid_fps
                gc.collect() 

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
                
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=1)
                rf.fit(X_train, y_train)
                
                st.session_state.model = rf
                st.success(f"训练完成！测试集 AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")
                
                # 绘制简单的 ROC
                fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr)
                ax.plot([0,1],[0,1],'k--')
                st.pyplot(fig)
                
                del X, y, X_train, X_test
                gc.collect()

    elif menu == "分子设计":
        st.header("3. 分子生成")
        n_gen = st.slider("数量", 10, 50, 20)
        if st.button("生成"):
            seeds = st.session_state.active_seeds[:10]
            new_mols = generate_brics_molecules(seeds, n_gen=n_gen)
            st.session_state.generated_mols = new_mols
            mols = [Chem.MolFromSmiles(s) for s in new_mols if Chem.MolFromSmiles(s)]
            if mols:
                st.image(Draw.MolsToGridImage(mols[:12], molsPerRow=4))

    elif menu == "虚拟筛选(ADMET+对接)":
        st.header("4. 虚拟筛选")
        if not st.session_state.generated_mols or st.session_state.model is None:
            st.warning("请先生成分子并训练模型")
        else:
            if st.button("开始筛选"):
                results = []
                for smi in st.session_state.generated_mols:
                    prob = predict_activity(st.session_state.model, smi, st.session_state.fp_params)
                    dock = calculate_docking_score(smi)
                    results.append({"SMILES": smi, "活性概率": prob, "对接能量": dock})
                df_res = pd.DataFrame(results).sort_values("活性概率", ascending=False)
                st.dataframe(df_res)

    elif menu == "药物重定位":
        st.header("5. 药物重定位")
        if st.session_state.model:
            lib = get_drug_library()
            if st.button("预测"):
                lib["活性概率"] = lib["smiles"].apply(lambda s: predict_activity(st.session_state.model, s, st.session_state.fp_params))
                st.dataframe(lib.sort_values("活性概率", ascending=False))

    elif menu == "耐药性分析":
        st.header("6. 耐药性测试")
        if st.session_state.generated_mols and st.session_state.model:
            smi = st.selectbox("选择分子", st.session_state.generated_mols)
            fp = get_morgan_fp(smi)
            probs = []
            for _ in range(30):
                noise = np.random.binomial(1, 0.05, fp.shape)
                p = st.session_state.model.predict_proba(np.clip(fp + noise, 0, 1).reshape(1,-1))[0][1]
                probs.append(p)
            fig, ax = plt.subplots()
            ax.plot(probs)
            st.pyplot(fig)

    elif menu == "知识提取":
        st.header("7. 文献分析")
        key = st.text_input("OpenAI Key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            client = OpenAI()
            if st.button("搜索并分析"):
                ids = search_pubmed("PCSK9 inhibitor", 2)
                for i in ids:
                    t, a = fetch_abstract(i)
                    st.write(f"**{t}**")
                    st.write(a[:300] + "...")

    elif menu == "分子3D展示":
        st.header("8. 3D 可视化")
        smi = st.text_input("SMILES", "c1ccccc1NC(=O)C")
        if smi:
            display_molecule_3d(smi)

if __name__ == "__main__":
    main()
