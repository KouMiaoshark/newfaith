import json
import os
import sys
import torch
import yaml

# 1. 先添加项目根目录到Python路径
PROJECT_ROOT = "/home/koumiao/FAITH-main"
sys.path.append(PROJECT_ROOT)

# 2. 导入依赖（添加完路径后再导入）
from faith.library.utils import get_config, get_logger, store_json_with_mkdir
from faith.temporal_qu.seq2seq_tsf.seq2seq_tsf_model import Seq2SeqTSFModel


class Seq2SeqTSFModule:
    def __init__(self, config):
        """Initialize TSF module."""
        self.config = config
        self.logger = get_logger(__name__, config)

        # create model
        self.tsf_model = Seq2SeqTSFModel(config)
        self.model_loaded = False

        self.tsf_delimiter = config["tsf_delimiter"]

    def train(self):
        """Train the model on silver TSF data."""
        self.logger.info(f"Starting training...")
        benchmark = self.config["benchmark"]
        path_to_data = self.config["path_to_data"]
        data_dir = os.path.join(path_to_data, benchmark, self.config["tsf_annotated_path"])

        tsf_train_data = self.config["tsf_train"]
        tsf_dev_data = self.config["tsf_dev"]

        train_path = os.path.join(data_dir, tsf_train_data)
        dev_path = os.path.join(data_dir, tsf_dev_data)

        self.tsf_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def _load(self):
        """真实的模型加载逻辑"""
        if not self.model_loaded:
            self.tsf_model.load()  # 加载预训练模型权重
            self.tsf_model.set_eval_mode()  # 设置为评估模式
            self.model_loaded = True

    def _format_tsf(self, tsf, output_format="sequence"):
        """真实的TSF格式化逻辑"""
        if output_format == "sequence":
            slots = tsf.split(self.tsf_delimiter.strip(), 4)
            if len(slots) < 5:
                slots = slots + (5 - len(slots)) * [""]
            tsf = self.tsf_delimiter.join(slots)
        else:
            tsf = eval(tsf)
        return tsf

    def _inference(self, question):
        """真实的模型推理逻辑"""

        def _normalize_input(_input):
            return _input.replace(",", " ")

        def _normalize_tsf(tsf):
            tsf_slots = tsf.split(self.tsf_delimiter, 3)
            tsf = " ".join((tsf_slots[:1] + [tsf_slots[3]]))
            tsf = tsf.replace(",", " ").replace(self.tsf_delimiter.strip(), " ")
            return tsf

        def _hallucination(input_words, tsf_words):
            bools = [word in input_words for word in tsf_words]
            if False in bools:
                return True
            return False

        if self.config.get("tsf_avoid_hallucination"):
            tsfs = self.tsf_model.inference_top_k(question)
            tsfs = [self._format_tsf(tsf) for tsf in tsfs]

            input_words = _normalize_input(question).split()
            for tsf in tsfs:
                tsf_words = _normalize_tsf(tsf).split()
                if not _hallucination(input_words, tsf_words):
                    return tsf
            return tsfs[0]
        else:
            tsf = self.tsf_model.inference_top_1(question)
            tsf = self._format_tsf(tsf)
            return tsf

    def inference_on_instance(self, instance):
        """Run inference on a single question."""
        self._load()
        with torch.no_grad():
            # 兼容传入字符串（question）或字典（带Question字段）
            if isinstance(instance, str):
                input_question = instance
            else:
                input_question = instance.get("question") or instance.get("Question")

            # 调用真实的推理逻辑
            structured_temporal_form = self._inference(input_question)
            res_instance = {
                "question": input_question,
                "structured_temporal_form": structured_temporal_form
            }
            return res_instance

    def inference_on_question(self, question):
        """Run inference on a single question（返回纯TSF结果）"""
        return self.inference_on_instance({"question": question})["structured_temporal_form"]


def get_config(path):
    """Load the config dict from the given .yml file."""
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


# ===================== 核心配置 =====================
PROJECT_ROOT = "/home/koumiao/FAITH-main"
example_path = os.path.join(PROJECT_ROOT, "_data/tiq/tsf_annotation/simplify_rela_dev_solution1_bart.json")
output_path = os.path.join(PROJECT_ROOT, "_data/tiq/tsf_annotation/simplify_rela_dev_solution1_bart_1.json")
config_path = os.path.join(PROJECT_ROOT, "config/tiq/train_tqu_fer.yml")
save_batch_size = 10
# ====================================================

if __name__ == "__main__":
    # 1. 校验文件是否存在
    if not os.path.exists(example_path):
        print(f"❌ 错误：找不到文件 {example_path}")
        exit(1)
    if not os.path.exists(config_path):
        print(f"❌ 错误：找不到config文件 {config_path}")
        exit(1)

    # 2. 加载配置和数据
    config = get_config(config_path)
    try:
        with open(example_path, 'r', encoding="utf-8") as f:
            data_list = json.load(f)
        # 校验data_list的类型（必须是列表）
        if not isinstance(data_list, list):
            print(f"❌ 错误：JSON文件解析结果不是列表，而是 {type(data_list)}")
            exit(1)
        print(f"✅ 成功读取JSON文件，共 {len(data_list)} 条数据")
    except json.JSONDecodeError:
        print(f"❌ 错误：{example_path} 不是有效的JSON文件")
        exit(1)

    # 3. 初始化模型（只初始化一次）
    srm = Seq2SeqTSFModule(config)

    # 测试单个问题（验证模型）
    example = "what awards were gladys knight & the pips nominated for during wwii"
    res = srm.inference_on_question(example)
    print(f"✅ 单个问题推理结果：{res}")

    # 4. 批量处理文件数据
    processed_data = []
    error_data_indices = []  # 记录错误数据的下标，方便排查
    total_data = len(data_list)

    for idx, item in enumerate(data_list):
        try:
            # ========== 关键：添加数据类型和字段校验 ==========
            # 校验item是否是字典
            if not isinstance(item, dict):
                raise ValueError(f"数据类型错误，预期字典，实际是 {type(item)}，数据内容：{item}")
            # 校验是否包含Question字段
            if "Question" not in item:
                raise ValueError(f"字典中缺少 'Question' 字段，数据内容：{item.keys()}")

            # 获取问题文本（经过校验，这里不会报错）
            question = item["Question"]
            # 调用真实模型推理
            tsf_result = srm.inference_on_question(question)
            # 添加新字段
            item["solu1_model_rela"] = tsf_result
            processed_data.append(item)

            # 分批保存
            if (idx + 1) % save_batch_size == 0:
                with open(output_path, 'w', encoding="utf-8") as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                print(f"✅ 已保存 {idx + 1}/{total_data} 条数据")

        except Exception as e:
            error_info = f"第 {idx + 1} 条数据出错：{str(e)}"
            print(f"⚠️  {error_info}")
            error_data_indices.append((idx + 1, str(e)))  # 记录错误数据下标和原因
            # 出错时保存已处理数据
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            continue

    # 保存最后剩余数据
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 输出错误数据汇总（方便排查JSON文件问题）
    print(f"\n📊 处理完成汇总：")
    print(f"   - 总数据量：{total_data}")
    print(f"   - 成功处理：{len(processed_data)}")
    print(f"   - 错误数据量：{len(error_data_indices)}")
    if error_data_indices:
        print(f"   - 错误数据下标和原因：{error_data_indices[:10]}")  # 只显示前10条，避免刷屏
    print(f"🎉 结果已保存到 {output_path}")