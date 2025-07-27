import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict

def parse_drugbank_xml_fixed(xml_file_path):
    """
    修复版DrugBank解析器：正确的疾病映射和更高的覆盖率
    """
    print("步骤 1: 开始解析DrugBank XML文件（修复版）...")
    
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
    therapeutic_data = []
    
    # 统计信息
    stats = defaultdict(int)
    drugs_without_info = []
    
    for event, elem in tqdm(context, desc="Parsing DrugBank"):
        if event == 'end' and elem.tag == f"{{{namespace}}}drug":
            # 只处理小分子药物
            if elem.attrib.get('type') != 'small molecule':
                elem.clear()
                root.clear()
                continue
            
            # 基本信息
            db_id_node = elem.find('db:drugbank-id[@primary="true"]', ns_map)
            name_node = elem.find('db:name', ns_map)
            
            if db_id_node is None or name_node is None:
                elem.clear()
                root.clear()
                continue
            
            drugbank_id = db_id_node.text
            name = name_node.text
            
            # SMILES
            smiles_node = elem.find("db:calculated-properties/db:property[db:kind='SMILES']/db:value", ns_map)
            smiles = smiles_node.text if smiles_node is not None else None
            
            if not smiles:
                elem.clear()
                root.clear()
                continue
            
            # 描述
            descr_node = elem.find('db:description', ns_map)
            description = (
                ET.tostring(descr_node, method='text', encoding='unicode').strip()
                if descr_node is not None else ""
            )
            
            drugs_data.append({
                'drug_id': drugbank_id,
                'name': name,
                'smiles': smiles,
                'description': description
            })
            
            # 收集所有治疗相关信息
            drug_therapeutic_info = []
            
            # 1. 传统indication（最高优先级）
            indications_found = set()
            
            # 方法1: db:indications/db:indication
            indications_wrapper = elem.find('db:indications', ns_map)
            if indications_wrapper is not None:
                for ind in indications_wrapper.findall('db:indication', ns_map):
                    if ind.text and ind.text.strip():
                        indication_text = ind.text.strip()
                        # 从indication文本中提取疾病
                        diseases = extract_diseases_from_indication(indication_text)
                        for disease in diseases:
                            indications_found.add(disease)
                            drug_therapeutic_info.append({
                                'drug_id': drugbank_id,
                                'disease_name': disease,
                                'source': 'indication',
                                'confidence': 1.0
                            })
                        stats['indication'] += len(diseases)
            
            # 方法2: 直接的 db:indication
            for ind in elem.findall('db:indication', ns_map):
                if ind.text and ind.text.strip():
                    indication_text = ind.text.strip()
                    diseases = extract_diseases_from_indication(indication_text)
                    for disease in diseases:
                        if disease not in indications_found:
                            indications_found.add(disease)
                            drug_therapeutic_info.append({
                                'drug_id': drugbank_id,
                                'disease_name': disease,
                                'source': 'indication',
                                'confidence': 1.0
                            })
                            stats['indication'] += 1
            
            # 2. Categories (药物分类)
            categories_elem = elem.find('db:categories', ns_map)
            if categories_elem is not None:
                for cat in categories_elem.findall('db:category', ns_map):
                    cat_name = cat.find('db:category', ns_map)
                    if cat_name is not None and cat_name.text:
                        category_text = cat_name.text.strip()
                        diseases = convert_category_to_diseases(category_text)
                        for disease in diseases:
                            if disease and disease.lower() not in indications_found:
                                drug_therapeutic_info.append({
                                    'drug_id': drugbank_id,
                                    'disease_name': disease,
                                    'source': 'category',
                                    'confidence': 0.8
                                })
                                stats['category'] += 1
            
            # 3. ATC codes
            atc_codes_elem = elem.find('db:atc-codes', ns_map)
            if atc_codes_elem is not None:
                for atc in atc_codes_elem.findall('db:atc-code', ns_map):
                    # 获取所有level的信息
                    for level in atc:
                        level_tag = level.tag.replace(f"{{{namespace}}}", "")
                        if level.text and 'level' in level_tag:
                            atc_text = level.text.strip()
                            diseases = convert_atc_to_diseases(atc_text)
                            for disease in diseases:
                                if disease and disease.lower() not in indications_found:
                                    drug_therapeutic_info.append({
                                        'drug_id': drugbank_id,
                                        'disease_name': disease,
                                        'source': 'atc',
                                        'confidence': 0.7
                                    })
                                    stats['atc'] += 1
            
            # 4. 从描述中提取（如果还没有找到任何治疗信息）
            if not drug_therapeutic_info and description:
                diseases = extract_diseases_from_text(description[:500])  # 只看前500字符
                for disease in diseases:
                    drug_therapeutic_info.append({
                        'drug_id': drugbank_id,
                        'disease_name': disease,
                        'source': 'description',
                        'confidence': 0.5
                    })
                    stats['description'] += 1
            
            # 5. Groups (如果还是没有信息)
            if not drug_therapeutic_info:
                groups_elem = elem.find('db:groups', ns_map)
                if groups_elem is not None:
                    for group in groups_elem.findall('db:group', ns_map):
                        if group.text and group.text.strip() == 'approved':
                            # 给approved药物一个通用分类
                            drug_therapeutic_info.append({
                                'drug_id': drugbank_id,
                                'disease_name': 'general therapeutic use',
                                'source': 'group',
                                'confidence': 0.3
                            })
                            stats['group'] += 1
                            break
            
            # 去重并添加到总列表
            seen = set()
            for info in drug_therapeutic_info:
                key = (info['drug_id'], info['disease_name'].lower())
                if key not in seen:
                    seen.add(key)
                    therapeutic_data.append(info)
            
            # 记录没有找到信息的药物
            if not drug_therapeutic_info:
                drugs_without_info.append((drugbank_id, name))
            
            elem.clear()
            root.clear()
    
    df_drugs = pd.DataFrame(drugs_data)
    df_therapeutic = pd.DataFrame(therapeutic_data)
    
    # 打印统计信息
    print(f"\n解析完成！")
    print(f"找到 {len(df_drugs)} 个小分子药物")
    print(f"找到 {len(df_therapeutic)} 条治疗关系")
    
    print(f"\n治疗信息来源分布:")
    for source, count in stats.items():
        print(f"  {source}: {count}")
    
    # 统计有治疗信息的药物
    drugs_with_info = df_therapeutic['drug_id'].nunique()
    print(f"\n有治疗信息的药物: {drugs_with_info} ({drugs_with_info/len(df_drugs)*100:.1f}%)")
    
    # 统计每个药物的平均治疗关系数
    avg_relations = len(df_therapeutic) / drugs_with_info if drugs_with_info > 0 else 0
    print(f"平均每个药物的治疗关系数: {avg_relations:.2f}")
    
    # 显示一些没有找到信息的药物
    if drugs_without_info:
        print(f"\n没有找到治疗信息的药物数: {len(drugs_without_info)}")
        print("示例（前5个）:")
        for drug_id, name in drugs_without_info[:5]:
            print(f"  {drug_id}: {name}")
    
    return df_drugs, df_therapeutic


def extract_diseases_from_indication(indication_text):
    """
    从indication文本中提取具体的疾病名称
    """
    diseases = []
    text_lower = indication_text.lower()
    
    # 常见的indication模式
    patterns = [
        r'treatment of (.+?)(?:\.|,|;|$)',
        r'indicated for (.+?)(?:\.|,|;|$)',
        r'used for (.+?)(?:\.|,|;|$)',
        r'management of (.+?)(?:\.|,|;|$)',
        r'therapy for (.+?)(?:\.|,|;|$)',
        r'prevention of (.+?)(?:\.|,|;|$)',
        r'prophylaxis of (.+?)(?:\.|,|;|$)',
        r'relief of (.+?)(?:\.|,|;|$)',
    ]
    
    # 提取疾病
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # 清理和分割多个疾病
            diseases_in_match = re.split(r' and | or |, ', match)
            for disease in diseases_in_match:
                disease = disease.strip()
                # 移除一些常见的修饰词
                disease = re.sub(r'^(the |acute |chronic |severe |mild |moderate )', '', disease)
                disease = re.sub(r'( in patients| in adults| in children).*$', '', disease)
                if len(disease) > 3 and disease not in ['use', 'users', 'patient', 'patients']:
                    diseases.append(disease)
    
    # 如果没有找到，返回简化的完整文本
    if not diseases and len(indication_text) < 100:
        # 移除常见的前缀
        simplified = re.sub(r'^(For |Used for |Indicated for |Treatment of )', '', indication_text)
        simplified = simplified.strip().rstrip('.')
        if len(simplified) > 3:
            diseases.append(simplified)
    
    return diseases[:3]  # 最多返回3个疾病


def convert_category_to_diseases(category):
    """
    将药物类别转换为疾病列表（修复版）
    """
    # 更准确的类别映射
    category_mapping = {
        # 抗感染类
        'Anti-Bacterial Agents': ['bacterial infections'],
        'Antibiotics': ['bacterial infections'],
        'Antiviral Agents': ['viral infections'],
        'Antifungal Agents': ['fungal infections'],
        'Antiparasitic Products': ['parasitic infections'],
        'Anti-Infective Agents': ['infections'],
        'Antimalarials': ['malaria'],
        'Antitubercular Agents': ['tuberculosis'],
        
        # 肿瘤类
        'Antineoplastic Agents': ['cancer'],
        'Antimetabolites': ['cancer'],
        'Antineoplastic and Immunomodulating Agents': ['cancer', 'immune disorders'],
        
        # 心血管类
        'Antihypertensive Agents': ['hypertension'],
        'Antiarrhythmic Agents': ['arrhythmia'],
        'Antianginal Agents': ['angina'],
        'Cardiovascular Agents': ['cardiovascular diseases'],
        'Vasodilator Agents': ['vascular disorders'],
        'Platelet Aggregation Inhibitors': ['thrombosis'],
        'Anticoagulants': ['blood clots', 'thrombosis'],
        'Antithrombotic Agents': ['thrombosis'],
        'Thrombin Inhibitors': ['thrombosis', 'blood clotting disorders'],
        'Factor Xa Inhibitors': ['thrombosis'],
        'Diuretics': ['fluid retention', 'hypertension'],
        'Beta Blocking Agents': ['hypertension', 'heart diseases'],
        'Calcium Channel Blockers': ['hypertension', 'angina'],
        'ACE Inhibitors': ['hypertension', 'heart failure'],
        'Angiotensin Receptor Blockers': ['hypertension'],
        
        # 神经精神类
        'Antidepressive Agents': ['depression'],
        'Antipsychotic Agents': ['psychosis', 'schizophrenia'],
        'Anti-Anxiety Agents': ['anxiety'],
        'Anxiolytics': ['anxiety'],
        'Hypnotics and Sedatives': ['insomnia'],
        'Anticonvulsants': ['epilepsy', 'seizures'],
        'Antiparkinsonian Agents': ['parkinson disease'],
        'Central Nervous System Agents': ['nervous system disorders'],
        'Analgesics': ['pain'],
        'Antimigraine Agents': ['migraine'],
        'Muscle Relaxants': ['muscle spasms', 'muscle disorders'],
        
        # 代谢内分泌类
        'Antidiabetic Agents': ['diabetes'],
        'Hypoglycemic Agents': ['diabetes'],
        'Thyroid Agents': ['thyroid disorders'],
        'Hormones': ['hormonal disorders'],
        'Corticosteroids': ['inflammatory conditions', 'autoimmune disorders'],
        'Sex Hormones': ['hormonal imbalances'],
        'Contraceptives': ['contraception'],
        'Anti-Obesity Agents': ['obesity'],
        'Lipid Regulating Agents': ['dyslipidemia', 'high cholesterol'],
        'Bone Density Conservation Agents': ['osteoporosis'],
        
        # 消化系统类
        'Gastrointestinal Agents': ['gastrointestinal disorders'],
        'Antiemetics': ['nausea', 'vomiting'],
        'Antacids': ['acid reflux', 'heartburn'],
        'Proton Pump Inhibitors': ['acid reflux', 'peptic ulcers'],
        'H2 Antagonists': ['acid reflux', 'ulcers'],
        'Laxatives': ['constipation'],
        'Antidiarrheals': ['diarrhea'],
        'Antiulcer Agents': ['ulcers'],
        
        # 呼吸系统类
        'Respiratory System Agents': ['respiratory diseases'],
        'Bronchodilator Agents': ['asthma', 'COPD'],
        'Antitussive Agents': ['cough'],
        'Expectorants': ['cough with mucus'],
        'Decongestants': ['nasal congestion'],
        'Antihistamines': ['allergies', 'allergic rhinitis'],
        
        # 炎症免疫类
        'Anti-Inflammatory Agents': ['inflammation'],
        'Immunosuppressive Agents': ['autoimmune diseases', 'organ rejection'],
        'Immunomodulatory Agents': ['immune system disorders'],
        'Antipyretics': ['fever'],
        'Antirheumatic Agents': ['rheumatoid arthritis'],
        
        # 皮肤类
        'Dermatologic Agents': ['skin conditions'],
        'Antipsoriatic Agents': ['psoriasis'],
        'Antiacne Agents': ['acne'],
        
        # 泌尿系统类
        'Urological Agents': ['urological disorders'],
        
        # 血液类
        'Hematologic Agents': ['blood disorders'],
        'Antianemic Agents': ['anemia'],
        
        # 其他
        'Enzyme Inhibitors': ['enzyme-related disorders'],
        'Vitamins': ['vitamin deficiency'],
        'Minerals': ['mineral deficiency'],
        'Nutritional Support': ['nutritional deficiency'],
        'Chelating Agents': ['heavy metal poisoning'],
        'Antiseptics': ['wound infections'],
        'Disinfectants': ['surface disinfection'],
        'Anesthetics': ['anesthesia', 'pain during procedures'],
    }
    
    # 先尝试完全匹配
    if category in category_mapping:
        return category_mapping[category]
    
    # 尝试部分匹配
    category_lower = category.lower()
    diseases = []
    
    for key, value in category_mapping.items():
        key_lower = key.lower()
        # 检查是否包含关键词
        if key_lower in category_lower or category_lower in key_lower:
            diseases.extend(value)
    
    # 如果还没找到，尝试更通用的规则
    if not diseases:
        # 处理包含"Agents"的类别
        if 'agents' in category_lower:
            # 提取关键词
            if 'anti' in category_lower:
                match = re.search(r'anti[- ]?(\w+)', category_lower)
                if match:
                    target = match.group(1)
                    if target not in ['agents', 'drug', 'drugs']:
                        diseases.append(f"{target}")
            
        # 处理"inhibitors"
        elif 'inhibitors' in category_lower:
            match = re.search(r'(\w+)\s+inhibitors', category_lower)
            if match:
                target = match.group(1)
                diseases.append(f"{target}-related disorders")
        
        # 处理"blockers"
        elif 'blockers' in category_lower:
            match = re.search(r'(\w+)\s+blockers', category_lower)
            if match:
                target = match.group(1)
                diseases.append(f"{target}-related disorders")
    
    return diseases


def convert_atc_to_diseases(atc_text):
    """
    将ATC分类转换为疾病列表（修复版）
    """
    atc_text_lower = atc_text.lower()
    
    # 更全面的ATC映射
    atc_mapping = {
        # 主要系统
        'alimentary tract and metabolism': ['gastrointestinal disorders', 'metabolic disorders'],
        'blood and blood forming organs': ['blood disorders', 'anemia'],
        'cardiovascular system': ['cardiovascular diseases'],
        'dermatologicals': ['skin diseases'],
        'genito urinary system and sex hormones': ['genitourinary disorders', 'hormonal disorders'],
        'systemic hormonal preparations': ['hormonal disorders'],
        'antiinfectives for systemic use': ['infections'],
        'antineoplastic and immunomodulating agents': ['cancer', 'immune disorders'],
        'musculo-skeletal system': ['musculoskeletal disorders'],
        'nervous system': ['neurological disorders'],
        'antiparasitic products': ['parasitic infections'],
        'respiratory system': ['respiratory diseases'],
        'sensory organs': ['eye diseases', 'ear disorders'],
        
        # 更具体的子分类
        'antibacterials for systemic use': ['bacterial infections'],
        'antimycotics for systemic use': ['fungal infections'],
        'antivirals for systemic use': ['viral infections'],
        'vaccines': ['disease prevention'],
        'antidiabetes drugs': ['diabetes'],
        'psycholeptics': ['psychiatric disorders', 'anxiety', 'insomnia'],
        'psychoanaleptics': ['depression', 'ADHD'],
        'analgesics': ['pain'],
        'anesthetics': ['anesthesia'],
        'antiepileptics': ['epilepsy'],
        'anti-parkinson drugs': ['parkinson disease'],
        'drugs used in addictive disorders': ['addiction'],
        'cardiac therapy': ['heart diseases'],
        'antihypertensives': ['hypertension'],
        'diuretics': ['fluid retention', 'hypertension'],
        'peripheral vasodilators': ['peripheral vascular disease'],
        'vasoprotectives': ['vascular disorders'],
        'beta blocking agents': ['hypertension', 'heart diseases'],
        'calcium channel blockers': ['hypertension', 'angina'],
        'agents acting on the renin-angiotensin system': ['hypertension', 'heart failure'],
        'lipid modifying agents': ['dyslipidemia', 'high cholesterol'],
        'antithrombotic agents': ['thrombosis', 'stroke prevention'],
        'antihemorrhagics': ['bleeding disorders'],
        'corticosteroids for systemic use': ['inflammatory conditions', 'autoimmune disorders'],
        'thyroid therapy': ['thyroid disorders'],
        'calcium homeostasis': ['calcium disorders', 'osteoporosis'],
        'insulins and analogues': ['diabetes'],
        'blood glucose lowering drugs': ['diabetes'],
        'sex hormones and modulators': ['hormonal disorders', 'reproductive disorders'],
        'urologicals': ['urological disorders', 'prostate disorders'],
        'drugs for obstructive airway diseases': ['asthma', 'COPD'],
        'cough and cold preparations': ['cough', 'cold symptoms'],
        'antihistamines for systemic use': ['allergies'],
        'ophthalmologicals': ['eye diseases'],
        'otologicals': ['ear disorders'],
        
        # 治疗用途
        'antacids': ['acid reflux', 'heartburn'],
        'drugs for peptic ulcer': ['peptic ulcers'],
        'antiemetics and antinauseants': ['nausea', 'vomiting'],
        'bile and liver therapy': ['liver disorders'],
        'laxatives': ['constipation'],
        'antidiarrheals': ['diarrhea'],
        'anti-inflammatory and antirheumatic products': ['inflammation', 'rheumatic disorders'],
        'topical products for joint and muscular pain': ['joint pain', 'muscle pain'],
        'antigout preparations': ['gout'],
        'drugs for treatment of bone diseases': ['bone diseases', 'osteoporosis'],
        'muscle relaxants': ['muscle spasms'],
        'antifungals for dermatological use': ['fungal skin infections'],
        'emollients and protectives': ['dry skin', 'skin protection'],
        'antipsoriatics': ['psoriasis'],
        'antibiotics and chemotherapeutics for dermatological use': ['skin infections'],
        'corticosteroids, dermatological preparations': ['skin inflammation'],
        'antiseptics and disinfectants': ['infections', 'wound care'],
        'medicated dressings': ['wound healing'],
        'antipruritics': ['itching'],
        'antiacne preparations': ['acne'],
    }
    
    diseases = []
    
    # 尝试匹配
    for key, value in atc_mapping.items():
        if key in atc_text_lower:
            diseases.extend(value)
    
    # 如果没有直接匹配，尝试提取关键信息
    if not diseases:
        # 处理"therapy"模式
        if 'therapy' in atc_text_lower:
            match = re.search(r'(\w+)\s+therapy', atc_text_lower)
            if match:
                organ = match.group(1)
                if organ not in ['drug', 'drugs']:
                    diseases.append(f"{organ} disorders")
        
        # 处理"drugs for"模式
        elif 'drugs for' in atc_text_lower:
            match = re.search(r'drugs for (.+)', atc_text_lower)
            if match:
                condition = match.group(1).strip()
                diseases.append(condition)
        
        # 处理"anti-"前缀
        elif 'anti' in atc_text_lower:
            match = re.search(r'anti[- ]?(\w+)', atc_text_lower)
            if match:
                target = match.group(1)
                if target.endswith('s'):
                    target = target[:-1]  # 移除复数
                diseases.append(target)
    
    # 去重
    return list(set(diseases))


def extract_diseases_from_text(text):
    """
    从文本中提取疾病名称（改进版）
    """
    diseases = []
    text_lower = text.lower()
    
    # 疾病提取模式
    patterns = [
        r'treatment of ([^,\.;]+)',
        r'indicated for ([^,\.;]+)',
        r'used for ([^,\.;]+)',
        r'effective against ([^,\.;]+)',
        r'management of ([^,\.;]+)',
        r'therapy for ([^,\.;]+)',
        r'prevention of ([^,\.;]+)',
        r'relief of ([^,\.;]+)',
        r'treatment and control of ([^,\.;]+)',
        r'prophylaxis of ([^,\.;]+)',
    ]
    
    # 常见疾病关键词
    disease_keywords = [
        'infection', 'cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia',
        'disease', 'disorder', 'syndrome', 'condition',
        'pain', 'inflammation', 'fever', 'allergy', 'allergies',
        'diabetes', 'hypertension', 'depression', 'anxiety', 'psychosis',
        'epilepsy', 'seizure', 'asthma', 'arthritis', 'osteoporosis',
        'migraine', 'headache', 'insomnia', 'obesity',
        'anemia', 'pneumonia', 'bronchitis', 'gastritis', 'colitis',
        'hepatitis', 'nephritis', 'dermatitis', 'rhinitis',
        'sinusitis', 'pharyngitis', 'meningitis', 'encephalitis',
        'thrombosis', 'embolism', 'ischemia', 'hemorrhage', 'stroke',
        'infarction', 'failure', 'insufficiency', 'deficiency',
        'ulcer', 'reflux', 'nausea', 'vomiting', 'diarrhea', 'constipation',
        'acne', 'psoriasis', 'eczema', 'rash',
        'angina', 'arrhythmia', 'fibrillation',
        'cough', 'cold', 'flu', 'influenza',
        'malaria', 'tuberculosis', 'HIV', 'AIDS'
    ]
    
    # 使用模式提取
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            match = match.strip()
            # 清理
            match = re.sub(r'^(the |acute |chronic |severe |mild |moderate )', '', match)
            match = re.sub(r'( in patients| in adults| in children).*$', '', match)
            
            # 检查是否包含疾病关键词
            if any(keyword in match for keyword in disease_keywords):
                if len(match) > 3 and match not in diseases:
                    diseases.append(match)
    
    # 如果没有找到，查找包含疾病关键词的短语
    if not diseases:
        for keyword in disease_keywords[:30]:  # 只用最常见的30个
            if keyword in text_lower:
                # 找到关键词的位置
                index = text_lower.find(keyword)
                # 提取周围的词
                start = max(0, index - 30)
                end = min(len(text_lower), index + len(keyword) + 30)
                context = text_lower[start:end]
                
                # 提取包含关键词的短语
                words = context.split()
                for i, word in enumerate(words):
                    if keyword in word:
                        # 获取前后的词
                        phrase_start = max(0, i - 2)
                        phrase_end = min(len(words), i + 3)
                        phrase = ' '.join(words[phrase_start:phrase_end])
                        
                        # 清理
                        phrase = re.sub(r'[^\w\s-]', '', phrase)
                        phrase = phrase.strip()
                        
                        if len(phrase) > len(keyword) and phrase not in diseases:
                            diseases.append(phrase)
                            break
    
    # 去重和清理
    unique_diseases = []
    seen = set()
    for disease in diseases:
        disease_clean = disease.strip().lower()
        # 移除太短或太长的
        if 3 < len(disease_clean) < 50 and disease_clean not in seen:
            seen.add(disease_clean)
            unique_diseases.append(disease_clean)
    
    return unique_diseases[:3]  # 最多返回3个


# 使用示例
if __name__ == '__main__':
    df_drugs, df_therapeutic = parse_drugbank_xml_fixed('drugbank.xml')
    
    # 保存结果
    df_drugs.to_csv('drugs_fixed.csv', index=False)
    df_therapeutic.to_csv('drug_disease_relations_fixed.csv', index=False)
    
    print("\n数据已保存到:")
    print("  - drugs_fixed.csv")
    print("  - drug_disease_relations_fixed.csv")
    
    # 展示一些示例
    print("\n治疗关系示例（前30条）:")
    sample = df_therapeutic.head(30)
    for _, row in sample.iterrows():
        print(f"  {row['drug_id']} -> {row['disease_name']} (来源: {row['source']}, 置信度: {row['confidence']})")
    
    # 统计每个置信度级别的数量
    print("\n置信度分布:")
    confidence_counts = df_therapeutic['confidence'].value_counts().sort_index(ascending=False)
    for conf, count in confidence_counts.items():
        print(f"  {conf}: {count} ({count/len(df_therapeutic)*100:.1f}%)")