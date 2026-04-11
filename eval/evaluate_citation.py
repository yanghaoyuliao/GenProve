import os
import json
import glob
import re
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional


try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("错误：未找到 'sentence_transformers'。请运行 pip install sentence-transformers")
    exit()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型路径
RELEVANCE_MODEL_PATH = "/usr/data/wjx/trove/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class StandaloneCitationEvaluator:
    """
    独立的引文评估器。
    逻辑复刻自 CitationF1RewardModel_v2.py (Ref -> Gen 覆盖率模式)。
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # 正则表达式初始化
        # 3个捕获组: (id), (id), (relation) - 使用非贪婪匹配以防包含双引号
        self.tuple_pattern_3_groups_str = r'\(\"(\d+)\"\,\s*\"\d+\"\,\s*\"([^"]*)\"\)'
        # 注意：上面的正则为了健壮性，我使用了您确认过的 ([^"]*) 版本
        self.tuple_extractor_3_groups = re.compile(r'\(\"(\d+)\"\,\s*\"(\d+)\"\,\s*\"(.*?)\"\)')

        self._init_model()

    def _init_model(self):
        """初始化相关性评估模型"""
        logger.info(f"正在加载相关性模型: {RELEVANCE_MODEL_PATH} ...")
        try:
            self.relevance_model = SentenceTransformer(RELEVANCE_MODEL_PATH, device=self.device)
            logger.info("相关性模型加载成功。")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            # 如果本地路径失败，尝试从 HuggingFace 加载通用模型作为备选
            # self.relevance_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
            raise e

    def get_content_after_think_tag(self, text: str) -> str:
        """获取 '</think>' 标签之后的内容。"""
        if not text: return ""
        parts = text.split('</think>', 1)
        if len(parts) > 1:
            return parts[1].strip()
        else:
            return text.strip()

    def split_by_prove_tag(self, text: str) -> List[str]:
        """将文本按照[PROVE]标签分割成列表"""
        if not text: return []
        pattern = r'(\[PROVE:\s*[^]]*\])'
        pieces = re.split(pattern, text, flags=re.DOTALL)
        sentences = []
        if not pieces: return []
        i = 0
        while i < len(pieces):
            text_part = pieces[i]
            tag_part = ""
            if i + 1 < len(pieces) and pieces[i + 1].startswith('[PROVE:'):
                tag_part = pieces[i + 1]
                i += 2
            else:
                i += 1
            combined_part = (text_part + tag_part).strip()
            if combined_part:
                sentences.append(combined_part)
        return sentences

    def parse_generation(self, text: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        解析包含多个[PROVE]标记的文本
        使用健壮的 '先找块，再找元组' 逻辑
        """
        generated_text = re.sub(r'\[PROVE:\s*.*?\]', '', text, flags=re.DOTALL).strip()
        citations = []

        # 查找所有 PROVE 块
        citation_pattern = r'\[PROVE:([^]]*)\]'
        for match in re.finditer(citation_pattern, text, flags=re.DOTALL):
            citation_block = match.group(1)
            # 在块内查找所有元组
            for tuple_match in re.finditer(self.tuple_extractor_3_groups, citation_block):
                try:
                    citations.append({
                        "doc_id": int(tuple_match.group(1)),
                        "sent_id": int(tuple_match.group(2)),
                        "relation": tuple_match.group(3)
                    })
                except ValueError:
                    continue
        return generated_text, citations

    def compute_scores(self, gen_citations: List[Dict], ref_citations: List[Dict]) -> Tuple[float, float, float]:
        """
        计算 Precision, Recall, F1
        """
        gen_triples = set((c['doc_id'], c['sent_id'], c['relation']) for c in gen_citations)
        ref_triples = set((c['doc_id'], c['sent_id'], c['relation']) for c in ref_citations)

        if not ref_triples:
            # 参考为空：如果生成也为空，满分；否则0分
            score = 1.0 if not gen_triples else 0.0
            return score, score, score

        if not gen_triples:
            return 0.0, 0.0, 0.0

        correctly_cited_count = len(gen_triples.intersection(ref_triples))

        precision = correctly_cited_count / len(gen_triples)
        recall = correctly_cited_count / len(ref_triples)

        if (precision + recall) == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1_score

    def evaluate_sample(self, completion_text: str, reference_text: str) -> Dict[str, float]:
        """
        核心评估逻辑 (Ref -> Gen)
        """
        # 0. 预处理
        gen_content = self.get_content_after_think_tag(completion_text)
        gen_sentences = self.split_by_prove_tag(gen_content)
        ref_sentences = self.split_by_prove_tag(reference_text)

        # 1. 边界检查
        if not ref_sentences:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        if not gen_sentences:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        # 2. 预处理生成句子 (Embedding)
        gen_data = []
        gen_pure_texts = []
        for s in gen_sentences:
            pt, cites = self.parse_generation(s)
            gen_pure_texts.append(pt or "")
            gen_data.append({"text": pt, "citations": cites})

        gen_embeddings = None
        if any(gen_pure_texts):
            gen_embeddings = self.relevance_model.encode(gen_pure_texts, convert_to_tensor=True)

        sample_p, sample_r, sample_f1 = [], [], []

        # 3. 遍历参考句子 (Ref -> Gen 逻辑)
        for ref_sentence in ref_sentences:
            ref_pure_text, ref_citations = self.parse_generation(ref_sentence)

            max_sim_score = -1.0
            best_gen_citations = []

            # 寻找最佳匹配
            if gen_embeddings is not None and ref_pure_text and any(gen_pure_texts):
                ref_embedding = self.relevance_model.encode(ref_pure_text, convert_to_tensor=True)

                # 计算相似度
                cosine_scores = util.pytorch_cos_sim(ref_embedding, gen_embeddings).squeeze()

                # 处理 Tensor 维度
                if cosine_scores.numel() > 0:
                    if cosine_scores.dim() == 0:
                        max_sim_idx = 0
                        max_sim_score = cosine_scores.item()
                    else:
                        max_sim_idx = torch.argmax(cosine_scores).item()
                        max_sim_score = cosine_scores[max_sim_idx].item()

                    best_gen_citations = gen_data[max_sim_idx]["citations"]

            # 4. 阈值判断 (Hardcoded 0.50 as per v2)
            p, r, f1 = 0.0, 0.0, 0.0

            if max_sim_score >= 0.50:
                p, r, f1 = self.compute_scores(best_gen_citations, ref_citations)

            sample_p.append(p)
            sample_r.append(r)
            sample_f1.append(f1)

        # 5. 平均
        count = len(ref_sentences)
        return {
            "f1": sum(sample_f1) / count,
            "precision": sum(sample_p) / count,
            "recall": sum(sample_r) / count
        }


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def extract_content(item: Dict) -> Tuple[str, str]:
    """
    智能提取生成内容和参考内容。
    处理 messages 格式和 response/label/labels 格式。
    """
    comp = ""
    ref = ""

    # 1. 尝试提取生成内容 (Generation)
    comp = item.get('response') or item.get('completion') or item.get('output') or ""

    # 2. 尝试提取参考内容 (Reference)
    # [关键修复] 增加了 'labels' (复数)
    ref = (item.get('reference') or
           item.get('answer') or
           item.get('label') or
           item.get('labels') or  # <--- 处理 labels 字段
           "")

    # 3. 如果没找到 ref，尝试从 messages 中提取 (SWIFT 格式)
    if not ref and 'messages' in item and isinstance(item['messages'], list):
        # 倒序查找，找到最后一个 role 为 assistant 的消息作为参考答案
        for msg in reversed(item['messages']):
            if msg.get('role') == 'assistant':
                ref = msg.get('content', "")
                break

    # 4. 处理列表嵌套的情况 (防御性编程)
    if isinstance(comp, list): comp = comp[0]
    if isinstance(ref, list): ref = ref[0]

    return comp, ref


# ==========================================
# 主执行流程
# ==========================================

def main():
    # 配置
    INPUT_DIR = r"/usr/data/wjx/trove/eval/other_models_infer_data"
    OUTPUT_FILE = r"citation_evaluation_report.md"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(INPUT_DIR):
        logger.error(f"路径不存在: {INPUT_DIR}")
        return

    # 初始化评估器
    logger.info(f"在 {DEVICE} 上初始化评估器...")
    evaluator = StandaloneCitationEvaluator(device=DEVICE)

    # 获取文件
    jsonl_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    if not jsonl_files:
        logger.warning("未找到 .jsonl 文件。")
        return

    all_dataset_stats = []

    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        logger.info(f"正在处理: {file_name}")

        data = load_jsonl(file_path)
        metrics_list = {"f1": [], "precision": [], "recall": []}
        
        valid_samples = 0
        skipped_samples = 0

        for item in tqdm(data, desc=f"Evaluating {file_name}", leave=False):
            # [关键修改] 使用智能提取函数
            comp, ref = extract_content(item)

            if not comp or not ref:
                skipped_samples += 1
                continue
            
            valid_samples += 1
            scores = evaluator.evaluate_sample(comp, ref)
            metrics_list["f1"].append(scores["f1"])
            metrics_list["precision"].append(scores["precision"])
            metrics_list["recall"].append(scores["recall"])

        # 计算数据集均值
        if metrics_list["f1"]:
            stats = {
                "Dataset": file_name,
                "Samples": len(metrics_list["f1"]),
                "F1": np.mean(metrics_list["f1"]) * 100,
                "Precision": np.mean(metrics_list["precision"]) * 100,
                "Recall": np.mean(metrics_list["recall"]) * 100
            }
            all_dataset_stats.append(stats)
            logger.info(f"结果 {file_name}: F1={stats['F1']:.2f}% (有效样本: {valid_samples}, 跳过: {skipped_samples})")
        else:
            logger.warning(f"数据集 {file_name} 只有 {valid_samples} 个有效样本 (跳过 {skipped_samples})，无法计算统计。")

    # 生成报告
    if all_dataset_stats:
        df = pd.DataFrame(all_dataset_stats)
        df = df.sort_values(by="F1", ascending=False)

        # 添加总平均行
        avg_row = {
            "Dataset": "**AVERAGE**",
            "Samples": df["Samples"].sum(),
            "F1": df["F1"].mean(),
            "Precision": df["Precision"].mean(),
            "Recall": df["Recall"].mean()
        }
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        # 写入 Markdown
        md_text = f"# 🚀 引文评估报告 (Citation Evaluation)\n\n"
        md_text += f"- **Date**: {pd.Timestamp.now()}\n"
        md_text += f"- **Path**: `{INPUT_DIR}`\n"
        md_text += f"- **Strategy**: Reference -> Generation Coverage (Threshold: 0.5)\n\n"
        md_text += df.to_markdown(index=False, floatfmt=".2f")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(md_text)

        print("\n" + "=" * 50)
        print("FINAL REPORT")
        print("=" * 50)
        print(df.to_string(index=False))
        print(f"\n报告已保存至: {os.path.abspath(OUTPUT_FILE)}")
    else:
        logger.warning("没有生成任何统计数据。")


if __name__ == "__main__":
    main()