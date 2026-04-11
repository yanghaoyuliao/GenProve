import re
import os
import json
import glob
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    文本清理工具类，专门用于清理包含 think 标签和 PROVE 标签的文本
    """

    def __init__(self):
        # 定义用于清理的正则表达式模式
        self.think_pattern = re.compile(r'</?think>', flags=re.DOTALL)
        self.no_think_pattern = re.compile(r'/no_think')
        self.prove_pattern = re.compile(r'\s*\[PROVE:.*?\]', flags=re.DOTALL)

    def remove_think_tags(self, text: str) -> str:
        """
        移除 <think> </think> 和 /no_think 标签
        """
        if not isinstance(text, str):
            return ""

        text = self.think_pattern.sub('', text)
        text = self.no_think_pattern.sub('', text)
        return text

    def remove_prove_tags(self, text: str) -> str:
        """
        移除 [PROVE: ...] 标注
        """
        if not isinstance(text, str):
            return ""

        text = self.prove_pattern.sub('', text)
        return text

    def clean_text(self, text: str, remove_think: bool = True, remove_prove: bool = True) -> str:
        """
        综合清理文本
        """
        if not isinstance(text, str):
            return ""

        cleaned_text = text

        if remove_think:
            cleaned_text = self.remove_think_tags(cleaned_text)

        if remove_prove:
            cleaned_text = self.remove_prove_tags(cleaned_text)

        cleaned_text = cleaned_text.strip()
        return cleaned_text


class StrictProveValidator:
    """
    严格的PROVE内容验证器
    """

    def __init__(self):
        # 定义允许的关系类型
        self.allowed_relations = {'Quotation', 'Compression', 'Inference', 'Other'}

        # 正确的完整格式模式 - 必须包含双引号
        self.correct_pattern = re.compile(
            r'\(\s*\"\d+\"\s*,\s*\"\d+\"\s*,\s*\"(Quotation|Compression|Inference|Other)\"\s*\)'
        )

        # 检测错误格式 - 缺少双引号
        self.incorrect_pattern = re.compile(
            r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(Quotation|Compression|Inference|Other)\s*\)'
        )

    def validate_prove_content(self, prove_content: str) -> Tuple[bool, List[str]]:
        """
        严格验证PROVE内容格式
        """
        errors = []

        # 检查是否有错误格式（缺少双引号）
        incorrect_matches = self.incorrect_pattern.findall(prove_content)
        if incorrect_matches:
            for doc_id, sent_id, relation in incorrect_matches:
                errors.append(
                    f"三元组格式错误: ({doc_id},{sent_id},{relation}) - 缺少双引号，正确格式应为: (\"{doc_id}\", \"{sent_id}\", \"{relation}\")")
            return False, errors

        # 检查正确格式
        correct_matches = self.correct_pattern.findall(prove_content)
        if not correct_matches:
            errors.append("未找到符合格式的三元组")

        return len(errors) == 0, errors


class TrainingResultValidator:
    """
    训练结果验证器，用于检查训练输出是否符合要求
    """

    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.prove_validator = StrictProveValidator()

        # 用于提取PROVE标签的正则表达式
        self.prove_pattern = re.compile(r'\[PROVE:\s*(.*?)\]', re.DOTALL)

    def validate_prove_format(self, text: str) -> Tuple[bool, List[str]]:
        """
        验证PROVE标签格式是否正确
        """
        errors = []

        # 查找所有PROVE标签
        prove_matches = self.prove_pattern.findall(text)

        if not prove_matches:
            errors.append("未找到任何PROVE标签")
            return False, errors

        for i, prove_content in enumerate(prove_matches):
            prove_content = prove_content.strip()

            if not prove_content:
                errors.append(f"第{i + 1}个PROVE标签内容为空")
                continue

            # 使用严格的验证器检查PROVE内容
            is_valid, content_errors = self.prove_validator.validate_prove_content(prove_content)
            if not is_valid:
                errors.extend([f"第{i + 1}个PROVE标签: {err}" for err in content_errors])

        return len(errors) == 0, errors

    def has_substantial_text_content(self, text: str) -> bool:
        """
        检查文本是否有实质性的内容（不仅仅是PROVE标签）
        """
        text_without_prove = self.text_cleaner.remove_prove_tags(text)
        text_without_tags = self.text_cleaner.remove_think_tags(text_without_prove)
        cleaned_text = text_without_tags.strip()
        return len(cleaned_text) >= 10

    def validate_response_content(self, response: str) -> Tuple[bool, List[str]]:
        """
        验证response内容是否合理
        """
        errors = []

        if not response or len(response.strip()) == 0:
            errors.append("response字段为空")
            return False, errors

        prove_valid, prove_errors = self.validate_prove_format(response)
        if not prove_valid:
            errors.extend(prove_errors)

        if not self.has_substantial_text_content(response):
            errors.append("response缺少实质性文本内容，只有PROVE标签")

        if re.search(r'\[PROVE:[^\]]*$', response):
            errors.append("存在不完整的PROVE标签")

        return len(errors) == 0, errors

    def extract_prove_statistics(self, text: str) -> Dict[str, Any]:
        """
        提取PROVE标签的统计信息
        """
        stats = {
            'total_prove_tags': 0,
            'total_citations': 0,
            'relation_counts': {},
            'has_incorrect_format': False,
            'format_errors': []
        }

        prove_matches = self.prove_pattern.findall(text)
        stats['total_prove_tags'] = len(prove_matches)

        for prove_content in prove_matches:
            is_valid, errors = self.prove_validator.validate_prove_content(prove_content)
            if not is_valid:
                stats['has_incorrect_format'] = True
                stats['format_errors'].extend(errors)

            # 统计正确格式的引用
            correct_matches = self.prove_validator.correct_pattern.findall(prove_content)
            stats['total_citations'] += len(correct_matches)

        return stats

    def validate_data_item(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证单个数据项
        """
        result = {
            'original_item': data_item,  # 保持原始数据不变
            'is_valid': True,
            'errors': [],
            'prove_stats': {}
        }

        try:
            if 'response' in data_item:
                response = data_item['response']
                response_valid, response_errors = self.validate_response_content(response)
                result['prove_stats'] = self.extract_prove_statistics(response)

                if not response_valid:
                    result['is_valid'] = False
                    result['errors'].extend([f"response错误: {err}" for err in response_errors])
            else:
                result['is_valid'] = False
                result['errors'].append("缺少response字段")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"验证过程中发生异常: {str(e)}")

        return result

    def validate_training_results(self, input_dir: str, output_dir: str, file_pattern: str = "gpt5_4096_infer.jsonl") -> Dict[
        str, Any]:
        """
        验证训练结果目录中的所有文件
        """
        os.makedirs(output_dir, exist_ok=True)

        valid_output_dir = os.path.join(output_dir, "valid_data")
        invalid_output_dir = os.path.join(output_dir, "invalid_data")
        os.makedirs(valid_output_dir, exist_ok=True)
        os.makedirs(invalid_output_dir, exist_ok=True)

        input_files = glob.glob(os.path.join(input_dir, file_pattern))

        logger.info(f"找到 {len(input_files)} 个文件需要验证")

        stats = {
            'total_files': len(input_files),
            'total_items': 0,
            'valid_items': 0,
            'invalid_items': 0,
            'prove_statistics': {
                'total_prove_tags': 0,
                'total_citations': 0,
                'items_with_incorrect_format': 0
            },
            'file_stats': {}
        }

        for input_file in tqdm(input_files, desc="验证文件"):
            filename = os.path.basename(input_file)
            file_stats = {
                'filename': filename,
                'total_items': 0,
                'valid_items': 0,
                'invalid_items': 0,
                'invalid_details': [],
                'prove_statistics': {
                    'total_prove_tags': 0,
                    'total_citations': 0,
                    'items_with_incorrect_format': 0
                }
            }

            valid_output_file = os.path.join(valid_output_dir, f"valid_{filename}")
            invalid_output_file = os.path.join(invalid_output_dir, f"invalid_{filename}")

            valid_items = []
            invalid_items = []

            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            file_stats['total_items'] += 1
                            stats['total_items'] += 1

                            validation_result = self.validate_data_item(data)

                            if 'response' in data:
                                prove_stats = self.extract_prove_statistics(data['response'])
                                file_stats['prove_statistics']['total_prove_tags'] += prove_stats['total_prove_tags']
                                file_stats['prove_statistics']['total_citations'] += prove_stats['total_citations']
                                if prove_stats['has_incorrect_format']:
                                    file_stats['prove_statistics']['items_with_incorrect_format'] += 1

                                stats['prove_statistics']['total_prove_tags'] += prove_stats['total_prove_tags']
                                stats['prove_statistics']['total_citations'] += prove_stats['total_citations']
                                if prove_stats['has_incorrect_format']:
                                    stats['prove_statistics']['items_with_incorrect_format'] += 1

                            if validation_result['is_valid']:
                                file_stats['valid_items'] += 1
                                stats['valid_items'] += 1
                                # 保存原始数据，不进行清理
                                valid_items.append(data)
                            else:
                                file_stats['invalid_items'] += 1
                                stats['invalid_items'] += 1
                                invalid_items.append({
                                    'line_number': line_number,
                                    'original_item': validation_result['original_item'],
                                    'errors': validation_result['errors'],
                                    'prove_stats': validation_result['prove_stats']
                                })
                                file_stats['invalid_details'].append({
                                    'line_number': line_number,
                                    'errors': validation_result['errors']
                                })

                        except json.JSONDecodeError:
                            logger.warning(f"文件 {filename} 第 {line_number} 行 JSON 格式无效")
                        except Exception as e:
                            logger.error(f"处理文件 {filename} 第 {line_number} 行时出错: {e}")

                if valid_items:
                    with open(valid_output_file, 'w', encoding='utf-8') as f:
                        for item in valid_items:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')

                if invalid_items:
                    with open(invalid_output_file, 'w', encoding='utf-8') as f:
                        for item in invalid_items:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')

                stats['file_stats'][filename] = file_stats
                logger.info(
                    f"文件 {filename} 验证完成: {file_stats['valid_items']} 有效, {file_stats['invalid_items']} 无效")

            except Exception as e:
                logger.error(f"处理文件 {filename} 时出错: {e}")

        stats_file = os.path.join(output_dir, "validation_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("\n" + "=" * 50)
        logger.info("验证完成总结:")
        logger.info(f"总文件数: {stats['total_files']}")
        logger.info(f"总数据项: {stats['total_items']}")
        logger.info(f"有效数据项: {stats['valid_items']} ({stats['valid_items'] / stats['total_items'] * 100:.2f}%)")
        logger.info( 
            f"无效数据项: {stats['invalid_items']} ({stats['invalid_items'] / stats['total_items'] * 100:.2f}%)")
        logger.info(f"总PROVE标签数: {stats['prove_statistics']['total_prove_tags']}")
        logger.info(f"总引用数: {stats['prove_statistics']['total_citations']}")
        logger.info(f"使用错误格式的数据项: {stats['prove_statistics']['items_with_incorrect_format']}")
        logger.info(f"详细统计已保存至: {stats_file}")
        logger.info("=" * 50)

        return stats

    def generate_validation_report(self, stats: Dict[str, Any], output_dir: str):
        """
        生成详细的验证报告
        """
        report_file = os.path.join(output_dir, "validation_report.md")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 训练结果验证报告\n\n")
            f.write("## 总体统计\n\n")
            f.write(f"- 总文件数: {stats['total_files']}\n")
            f.write(f"- 总数据项: {stats['total_items']}\n")
            f.write(
                f"- 有效数据项: {stats['valid_items']} ({stats['valid_items'] / stats['total_items'] * 100:.2f}%)\n")
            f.write(
                f"- 无效数据项: {stats['invalid_items']} ({stats['invalid_items'] / stats['total_items'] * 100:.2f}%)\n")
            f.write(f"- 总PROVE标签数: {stats['prove_statistics']['total_prove_tags']}\n")
            f.write(f"- 总引用数: {stats['prove_statistics']['total_citations']}\n")
            f.write(f"- 使用错误格式的数据项: {stats['prove_statistics']['items_with_incorrect_format']}\n\n")

            f.write("## 各文件详细统计\n\n")
            for filename, file_stat in stats['file_stats'].items():
                f.write(f"### {filename}\n\n")
                f.write(f"- 总项数: {file_stat['total_items']}\n")
                f.write(
                    f"- 有效项: {file_stat['valid_items']} ({file_stat['valid_items'] / file_stat['total_items'] * 100:.2f}%)\n")
                f.write(
                    f"- 无效项: {file_stat['invalid_items']} ({file_stat['invalid_items'] / file_stat['total_items'] * 100:.2f}%)\n")
                f.write(f"- PROVE标签数: {file_stat['prove_statistics']['total_prove_tags']}\n")
                f.write(f"- 引用数: {file_stat['prove_statistics']['total_citations']}\n")
                f.write(f"- 使用错误格式的项数: {file_stat['prove_statistics']['items_with_incorrect_format']}\n")

                if file_stat['invalid_details']:
                    error_counts = {}
                    for detail in file_stat['invalid_details']:
                        for error in detail['errors']:
                            error_type = error.split(':')[0] if ':' in error else error
                            error_counts[error_type] = error_counts.get(error_type, 0) + 1

                    f.write("\n- 常见错误类型:\n")
                    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  - {error_type}: {count} 次\n")

                f.write("\n")

            f.write("## 建议\n\n")
            if stats['invalid_items'] > 0:
                f.write("1. 检查无效数据项中的具体错误，修正PROVE标签格式\n")
                f.write(
                    "2. **确保所有PROVE标签使用正确的格式: `(\\\"doc_id\\\", \\\"sent_id\\\", \\\"relation\\\")`**\n")
                f.write("3. **特别注意: 不要使用 `(doc_id, sent_id, relation)` 这种缺少双引号的错误格式**\n")
                f.write("4. 确保关系类型使用允许的值: Quotation, Compression, Inference, Other\n")
                f.write("5. 确保response包含实质性文本内容，不能只有PROVE标签\n")
                f.write("6. 确保PROVE标签完整且正确闭合\n")
            else:
                f.write("所有数据项格式正确，无需修改。\n")

        logger.info(f"验证报告已保存至: {report_file}")


# 使用示例99
if __name__ == "__main__":
    # 初始化验证器
    validator = TrainingResultValidator()

    # 验证训练结果
    input_directory = r"E:\essay\trove\project\grpo\rebuild_reward\data\infer_result\final_train_data\infer_datasets"
    output_directory = "./Correct_format_v2"

    stats = validator.validate_training_results(input_directory, output_directory)

    # 生成详细报告
    validator.generate_validation_report(stats, output_directory)

    # 打印每个文件的详细统计
    print("\n各文件详细统计:")
    for filename, file_stat in stats['file_stats'].items():
        print(f"{filename}:")
        print(f"  总项数: {file_stat['total_items']}")
        print(
            f"  有效项: {file_stat['valid_items']} ({file_stat['valid_items'] / file_stat['total_items'] * 100:.2f}%)")
        print(
            f"  无效项: {file_stat['invalid_items']} ({file_stat['invalid_items'] / file_stat['total_items'] * 100:.2f}%)")
        print(f"  使用错误格式的项数: {file_stat['prove_statistics']['items_with_incorrect_format']}")

        if file_stat['invalid_details']:
            print("  前几个无效项的错误:")
            for i, detail in enumerate(file_stat['invalid_details'][:3]):
                print(f"    第{detail['line_number']}行: {', '.join(detail['errors'][:2])}")
                if i >= 2:
                    break
        print()