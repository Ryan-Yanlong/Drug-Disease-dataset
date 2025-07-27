import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import os
import sys
import numpy as np

# ==============================================================================
# 函数 1: 解析 DrugBank XML
# ==============================================================================
def parse_drugbank_xml(xml_file_path):
    print("步骤 1: 开始解析DrugBank XML文件...")
    try:
        context = ET.iterparse(xml_file_path, events=('start', 'end'))
        _, root = next(context)
    except ET.ParseError as e:
        print(f"XML 文件解析失败！错误: {e}")
        return pd.DataFrame(), pd.DataFrame()

    namespace = ''
    if '}' in root.tag:
        namespace = root.tag.split('}')[0][1:]
    ns_map = {'db': namespace}

    drugs_data = []
    indications_data = []

    for event, elem in tqdm(context, desc="Parsing DrugBank"):
        if event == 'end' and elem.tag == f"{{{namespace}}}drug":
            if elem.attrib.get('type') != 'small molecule':
                elem.clear(); root.clear(); continue
            db_id_node = elem.find('db:drugbank-id[@primary="true"]', ns_map)
            name_node  = elem.find('db:name', ns_map)
            if db_id_node is None or name_node is None:
                elem.clear(); root.clear(); continue
            drugbank_id = db_id_node.text
            name        = name_node.text

            smiles_node = elem.find("db:calculated-properties/db:property[db:kind='SMILES']/db:value", ns_map)
            smiles = smiles_node.text if smiles_node is not None else None
            if not smiles:
                elem.clear(); root.clear(); continue

            descr_node = elem.find('db:description', ns_map)
            description = (
                ET.tostring(descr_node, method='text', encoding='unicode').strip()
                if descr_node is not None else ""
            )
            drugs_data.append({'drug_id': drugbank_id, 'name': name, 'smiles': smiles, 'description': description})

            wrapper = elem.find('db:indications', ns_map)
            nodes = wrapper.findall('db:indication', ns_map) if wrapper is not None else elem.findall('db:indication', ns_map)
            for ind in nodes:
                if ind.text and ind.text.strip():
                    indications_data.append({'drug_id': drugbank_id, 'disease_name': ind.text.strip()})
            elem.clear(); root.clear()

    df_drugs = pd.DataFrame(drugs_data)
    df_inds  = pd.DataFrame(indications_data)
    print(f"解析完成！共找到 {len(df_drugs)} 个小分子药物；{len(df_inds)} 条 indication。")
    return df_drugs, df_inds

# ==============================================================================
# 函数 2: 解析 OFFSIDES CSV
# ==============================================================================
def parse_offsides_csv(offsides_file_path):
    print("\n步骤 2: 开始解析OFFSIDES CSV文件...")
    try:
        df = pd.read_csv(offsides_file_path, low_memory=False)
    except FileNotFoundError:
        print(f"错误：OFFSIDES 文件未找到：'{offsides_file_path}'")
        sys.exit(1)
    required = ['drug_concept_name', 'condition_concept_name']
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV 缺少必要列，需要 {required}，但实际有 {df.columns.tolist()}")
    df_se = df[required].dropna().drop_duplicates()
    df_se.columns = ['drug_name', 'disease_name']
    print(f"解析完成！共 {len(df_se)} 条药物-副作用关系。")
    return df_se

# ==============================================================================
# 主流程 - 生成完整关系矩阵
# ==============================================================================
if __name__ == '__main__':
    # —————————— 配置区 ——————————
    DRUG_XML     = 'drugbank.xml'
    OFFSIDES_CSV = 'offsides.csv'
    OUTPUT_DIR   = './drug_repositioning_dataset_complete'
    
    # 目标规模控制（可选）
    TARGET_DRUGS = 1500      # 目标药物数量
    TARGET_DISEASES = 1500   # 目标疾病数量
    
    # 采样策略
    PRIORITIZE_TREATMENT = True  # 是否优先保留有治疗关系的药物和疾病
    MIN_TREATMENT_RATIO = 0.1    # 最低治疗关系比例（如果太低会给出警告）
    # —————————— 结束配置 ——————————

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. & 2. 解析原始数据
    df_drugs_all, df_inds = parse_drugbank_xml(DRUG_XML)
    df_se_raw = parse_offsides_csv(OFFSIDES_CSV)

    # 3. 准备和标准化关系数据
    print("\n步骤 3: 准备和标准化关系数据...")
    
    # 标准化疾病名称
    df_inds['disease_name'] = df_inds['disease_name'].str.lower().str.strip()
    df_se_raw['disease_name'] = df_se_raw['disease_name'].str.lower().str.strip()
    
    # 将OFFSIDES的药物名映射到DrugBank ID
    name_to_id = df_drugs_all.set_index('name')['drug_id'].to_dict()
    lower_name_map = {name.lower(): name for name in name_to_id}
    df_se_raw['drug_name_lower'] = df_se_raw['drug_name'].str.lower().str.strip()
    df_se_raw['name'] = df_se_raw['drug_name_lower'].map(lower_name_map)
    df_se_raw.dropna(subset=['name'], inplace=True)
    df_se_raw['drug_id'] = df_se_raw['name'].map(name_to_id)
    df_side_effect = df_se_raw[['drug_id', 'disease_name']].copy()
    
    # 4. 收集所有独特的药物和疾病
    print("\n步骤 4: 收集所有独特的药物和疾病...")
    
    # 从两个数据源收集所有疾病
    all_diseases = set()
    all_diseases.update(df_inds['disease_name'].dropna().unique())
    all_diseases.update(df_side_effect['disease_name'].dropna().unique())
    
    # 从两个数据源收集所有药物（必须在DrugBank中且有SMILES）
    drugs_with_indications = set(df_inds['drug_id'].dropna().unique())
    drugs_with_side_effects = set(df_side_effect['drug_id'].dropna().unique())
    all_drug_ids = drugs_with_indications | drugs_with_side_effects
    
    # 确保所有药物都在DrugBank中且有SMILES
    all_drug_ids = all_drug_ids & set(df_drugs_all['drug_id'])
    
    print(f"总共找到 {len(all_drug_ids)} 个药物，{len(all_diseases)} 个疾病")
    
    # 5. 采样策略：优先保留治疗关系或随机采样
    print("\n步骤 5: 采样以达到目标规模...")
    
    if PRIORITIZE_TREATMENT:
        print("使用优先保留治疗关系的采样策略")
        
        # 首先，找出所有有治疗关系的药物和疾病
        drugs_with_treatment = set(df_inds['drug_id'].dropna().unique())
        diseases_with_treatment = set(df_inds['disease_name'].dropna().unique())
        
        print(f"有治疗关系的药物: {len(drugs_with_treatment)} 个")
        print(f"有治疗关系的疾病: {len(diseases_with_treatment)} 个")
        
        # 药物采样策略：优先保留有治疗关系的药物
        if TARGET_DRUGS and len(all_drug_ids) > TARGET_DRUGS:
            print(f"\n需要从 {len(all_drug_ids)} 个药物中采样 {TARGET_DRUGS} 个...")
            
            # 确保所有有治疗关系的药物都被保留
            selected_drugs = drugs_with_treatment & all_drug_ids
            print(f"  - 优先保留 {len(selected_drugs)} 个有治疗关系的药物")
            
            # 如果还需要更多药物，从剩余的药物中随机采样
            if len(selected_drugs) < TARGET_DRUGS:
                remaining_drugs = all_drug_ids - selected_drugs
                n_to_sample = min(TARGET_DRUGS - len(selected_drugs), len(remaining_drugs))
                additional_drugs = set(pd.Series(list(remaining_drugs)).sample(n=n_to_sample, random_state=42))
                selected_drugs = selected_drugs | additional_drugs
                print(f"  - 额外采样 {len(additional_drugs)} 个药物")
            
            all_drug_ids = selected_drugs
        
        # 疾病采样策略：优先保留有治疗关系的疾病
        if TARGET_DISEASES and len(all_diseases) > TARGET_DISEASES:
            print(f"\n需要从 {len(all_diseases)} 个疾病中采样 {TARGET_DISEASES} 个...")
            
            # 确保所有有治疗关系的疾病都被保留
            selected_diseases = diseases_with_treatment
            print(f"  - 优先保留 {len(selected_diseases)} 个有治疗关系的疾病")
            
            # 如果还需要更多疾病，从剩余的疾病中随机采样
            if len(selected_diseases) < TARGET_DISEASES:
                remaining_diseases = all_diseases - selected_diseases
                n_to_sample = min(TARGET_DISEASES - len(selected_diseases), len(remaining_diseases))
                additional_diseases = set(pd.Series(list(remaining_diseases)).sample(n=n_to_sample, random_state=42))
                selected_diseases = selected_diseases | additional_diseases
                print(f"  - 额外采样 {len(additional_diseases)} 个疾病")
            
            all_diseases = selected_diseases
    else:
        # 原始的随机采样策略
        print("使用随机采样策略")
        
        if TARGET_DRUGS and len(all_drug_ids) > TARGET_DRUGS:
            print(f"随机采样 {TARGET_DRUGS} 个药物...")
            all_drug_ids = set(pd.Series(list(all_drug_ids)).sample(n=TARGET_DRUGS, random_state=42))
        
        if TARGET_DISEASES and len(all_diseases) > TARGET_DISEASES:
            print(f"随机采样 {TARGET_DISEASES} 个疾病...")
            all_diseases = set(pd.Series(list(all_diseases)).sample(n=TARGET_DISEASES, random_state=42))
    
    # 6. 创建最终的实体列表
    print("\n步骤 6: 创建最终实体列表...")
    df_drugs_final = df_drugs_all[df_drugs_all['drug_id'].isin(all_drug_ids)].copy().reset_index(drop=True)
    df_diseases_final = pd.DataFrame({
        'disease_name': sorted(list(all_diseases))
    }).reset_index().rename(columns={'index': 'disease_id'})
    
    print(f"最终数据集: {len(df_drugs_final)} 个药物 × {len(df_diseases_final)} 个疾病 = {len(df_drugs_final) * len(df_diseases_final)} 个关系")
    
    # 预计算可保留的治疗关系数
    preserved_treatments = 0
    for _, row in df_inds.iterrows():
        if row['drug_id'] in all_drug_ids and row['disease_name'] in all_diseases:
            preserved_treatments += 1
    print(f"预计可保留的治疗关系: {preserved_treatments} 条（原始: {len(df_inds)} 条）")
    
    # 7. 构建完整的关系矩阵
    print("\n步骤 7: 构建完整的关系矩阵...")
    
    # 创建映射字典
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(df_drugs_final['drug_id'])}
    disease_to_idx = df_diseases_final.set_index('disease_name')['disease_id'].to_dict()
    
    # 初始化关系矩阵（默认为0）
    n_drugs = len(df_drugs_final)
    n_diseases = len(df_diseases_final)
    relation_matrix = np.zeros((n_drugs, n_diseases), dtype=np.int8)
    
    # 填充治疗关系（label = 1）
    treat_count = 0
    missing_drug_treat = 0
    missing_disease_treat = 0
    for _, row in df_inds.iterrows():
        if row['drug_id'] not in drug_to_idx:
            missing_drug_treat += 1
        elif row['disease_name'] not in disease_to_idx:
            missing_disease_treat += 1
        else:
            drug_idx = drug_to_idx[row['drug_id']]
            disease_idx = disease_to_idx[row['disease_name']]
            relation_matrix[drug_idx, disease_idx] = 1
            treat_count += 1
    
    print(f"治疗关系填充情况:")
    print(f"  - 成功填充: {treat_count} 条")
    print(f"  - 药物不在最终集合中: {missing_drug_treat} 条")
    print(f"  - 疾病不在最终集合中: {missing_disease_treat} 条")
    
    # 填充副作用关系（label = -1）
    se_count = 0
    conflict_count = 0
    for _, row in df_side_effect.iterrows():
        if row['drug_id'] in drug_to_idx and row['disease_name'] in disease_to_idx:
            drug_idx = drug_to_idx[row['drug_id']]
            disease_idx = disease_to_idx[row['disease_name']]
            # 如果已经标记为治疗关系，保持为1（治疗优先）
            if relation_matrix[drug_idx, disease_idx] == 1:
                conflict_count += 1
            elif relation_matrix[drug_idx, disease_idx] != -1:  # 避免重复
                relation_matrix[drug_idx, disease_idx] = -1
                se_count += 1
    
    print(f"\n副作用关系填充情况:")
    print(f"  - 成功填充: {se_count} 条")
    print(f"  - 与治疗关系冲突（保留治疗）: {conflict_count} 条")
    
    # 8. 将矩阵转换为交互格式的DataFrame
    print("\n步骤 8: 生成交互文件...")
    interactions_data = []
    for i in range(n_drugs):
        for j in range(n_diseases):
            interactions_data.append({
                'drug_id': df_drugs_final.iloc[i]['drug_id'],
                'disease_id': j,
                'label': relation_matrix[i, j]
            })
    
    df_interactions_final = pd.DataFrame(interactions_data)
    
    # 9. 统计与导出
    print("\n--- 数据集统计 ---")
    print(f"药物数量: {len(df_drugs_final)}")
    print(f"疾病数量: {len(df_diseases_final)}")
    print(f"总关系数: {len(df_interactions_final)} (完整矩阵)")
    print(f"\n标签分布:")
    counts = df_interactions_final['label'].value_counts().to_dict()
    total = len(df_interactions_final)
    for lbl in sorted(counts.keys()):
        cnt = counts.get(lbl, 0)
        print(f"  Label {lbl:>2}: {cnt:>8} 条 ({cnt/total:.2%})")
    
    print(f"\n实际填充的关系:")
    print(f"  治疗关系: {treat_count} 条")
    print(f"  副作用关系: {se_count} 条")
    
    # 警告检查
    non_zero_count = treat_count + se_count
    if non_zero_count > 0:
        treatment_ratio = treat_count / non_zero_count
        print(f"\n治疗关系占已知关系的比例: {treatment_ratio:.2%}")
        if treatment_ratio < MIN_TREATMENT_RATIO:
            print(f"⚠️  警告：治疗关系比例过低！建议检查数据源或调整采样策略。")
    
    print("--------------------")
    
    # 保存数据
    df_drugs_final.to_csv(os.path.join(OUTPUT_DIR, 'drugs.csv'), index=False)
    df_diseases_final.to_csv(os.path.join(OUTPUT_DIR, 'diseases.csv'), index=False)
    df_interactions_final.to_csv(os.path.join(OUTPUT_DIR, 'interactions.csv'), index=False)
    
    # 可选：也保存为稀疏格式（只保存非零关系）
    df_sparse = df_interactions_final[df_interactions_final['label'] != 0].copy()
    df_sparse.to_csv(os.path.join(OUTPUT_DIR, 'interactions_sparse.csv'), index=False)
    print(f"\n额外保存了稀疏格式（仅非零关系）: {len(df_sparse)} 条")
    
    # 可选：保存为矩阵格式
    matrix_df = pd.DataFrame(
        relation_matrix, 
        index=df_drugs_final['drug_id'], 
        columns=df_diseases_final['disease_name']
    )
    matrix_df.to_csv(os.path.join(OUTPUT_DIR, 'relation_matrix.csv'))
    print(f"保存了关系矩阵文件: {n_drugs}×{n_diseases}")

    print(f"\n完成！数据集保存在 '{OUTPUT_DIR}' 目录下。")