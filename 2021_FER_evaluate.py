import json
import copy
from faith.library.utils import get_config, store_json_with_mkdir
from faith.faithful_er.fer import FER

# =============================
# 运行范围控制：跑第 6~10 个问题
# =============================
START_ID = 6
END_ID = 11
SKIP_IDS = {2525}  # 明确跳过（你说这个ID缺失）；对 6~10 不影响


def load_rel_map(rel_json_path, key="Relation"):
    """
    rel_json_path: 你的提示词输出文件（JSON list）
    key: 你要用哪个字段做 relation（比如 "Relation" / "model_rela" / "solu1_model_rela"）
    返回: {Id: relation_str}
    """
    with open(rel_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rel_map = {}
    for item in data:
        qid = int(item["Id"])  # 允许 str/int
        rel_map[qid] = str(item[key]).strip()
    return rel_map


def filter_instances(base_instances, rel_map, min_id=1, max_id=3000, skip_ids=None):
    """
    只保留：
      1) min_id <= Id <= max_id
      2) 不在 skip_ids 中（比如 2525）
      3) 在 rel_map 中存在对应 relation（保证每条样本都被覆盖 relation，避免不公平）
      4) 必须包含 structured_temporal_form（否则 FER 不能跑）
    并按 Id 升序排序后返回
    """
    if skip_ids is None:
        skip_ids = set()

    filtered = []
    missing_stf = []
    for ins in base_instances:
        if "Id" not in ins:
            continue
        try:
            qid = int(ins["Id"])
        except Exception:
            continue

        # ✅ 范围过滤（关键改动）
        if qid < min_id or qid > max_id:
            continue

        if qid in skip_ids:
            continue
        if qid not in rel_map:
            continue
        if "structured_temporal_form" not in ins:
            missing_stf.append(qid)
            continue

        filtered.append(ins)

    if missing_stf:
        raise ValueError(
            f"以下 Id 缺 structured_temporal_form（BASE_TSF 可能不是TQU输出或被破坏）："
            f"{missing_stf[:20]} ... 共{len(missing_stf)}条"
        )

    filtered.sort(key=lambda x: int(x["Id"]))
    return filtered


def inject_relation(instances, rel_map):
    """
    把 rel_map 里的 relation 覆盖写入 structured_temporal_form["relation"]
    """
    changed = 0
    for ins in instances:
        qid = int(ins["Id"])
        ins["structured_temporal_form"]["relation"] = rel_map[qid]
        changed += 1
        print("覆盖成功", ins["structured_temporal_form"], "\n")
    return changed


def run_one(
    config_path,
    base_tsf_data_path,
    rel_json_path,
    rel_key,
    out_path,
    sources,
    min_id=1,
    max_id=3000,
    skip_ids=None,
):
    if skip_ids is None:
        skip_ids = set()

    config = get_config(config_path)
    fer = FER(config)

    # 先加载 relation 映射（因为要用它来过滤实例，保证公平）
    rel_map = load_rel_map(rel_json_path, key=rel_key)

    # 再加载 base TSF 数据
    with open(base_tsf_data_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    if not isinstance(base_data, list):
        raise ValueError("BASE_TSF 文件必须是 JSON list")

    # 深拷贝，避免污染原数据
    data = copy.deepcopy(base_data)

    # 过滤：只跑 min_id<=Id<=max_id；跳过 skip_ids；且必须在 rel_map 中有 relation
    data = filter_instances(data, rel_map, min_id=min_id, max_id=max_id, skip_ids=skip_ids)

    # 打印一下实际将要跑的范围与数量（注意：2525会被跳过）
    ids = [int(x["Id"]) for x in data]
    if ids:
        print(
            f"[filter] 将处理 {len(data)} 条：Id {min(ids)} ~ {max(ids)} "
            f"（range={min_id}..{max_id}，skip={sorted(list(skip_ids))}）"
        )
        if 2525 in ids:
            raise RuntimeError("逻辑错误：2525 不应出现在处理列表中")
    else:
        print("[filter] 过滤后没有任何样本可处理（检查 min_id/max_id/skip_ids/rel_map 是否匹配）")
        return

    # 覆盖 relation
    n_changed = inject_relation(data, rel_map)
    print(f"[inject] 覆盖 relation 数量 = {n_changed}（应等于处理样本数 {len(data)}）")

    # 跑 FER：会在每条样本里写入 answer_presence_initial / pruning / answer_presence 等字段
    fer.inference_on_data(data, sources)

    # 保存结果（JSON数组；虽然你扩展名写 .jsonl，但这里实际保存的是 JSON list）
    store_json_with_mkdir(data, out_path)

    # 用 FAITH 自带评估器输出统计（会生成 *.res 文件）
    fer.evaluate_retrieval_results_res_stage(out_path, stage="initial", sources=sources)
    fer.evaluate_retrieval_results_res_stage(out_path, stage="pruning", sources=sources)
    fer.evaluate_retrieval_results_res_stage(out_path, stage="scoring", sources=sources)

    fer.store_cache()
    print(f"[done] results saved to {out_path}")


if __name__ == "__main__":
    CONFIG = "config/tiq/evaluate.yml"  # 你的 config

    # 这份文件必须包含 structured_temporal_form（TQU 输出文件）
    BASE_TSF = "_intermediate_representations/tiq/seq2seq2_tqu/train_tqu-faith.json"

    REL_PROMPT1 = "_data/tiq/tsf_annotation/simplify_rela_train_solution1.json"
    REL_PROMPT2 = "_data/tiq/tsf_annotation/simplify_rela_train_solution2.json"

    REL_KEY = "Relation"
    SOURCES = ["kb", "text", "table", "info"]

    # 跑第 6~10 个问题
    run_one(
        CONFIG,
        BASE_TSF,
        REL_PROMPT1,
        REL_KEY,
        f"out/fer_prompt1_id_{START_ID}_{END_ID}.jsonl",
        SOURCES,
        min_id=START_ID,
        max_id=END_ID,
        skip_ids=SKIP_IDS,
    )

    run_one(
        CONFIG,
        BASE_TSF,
        REL_PROMPT2,
        REL_KEY,
        f"out/fer_prompt2_id_{START_ID}_{END_ID}.jsonl",
        SOURCES,
        min_id=START_ID,
        max_id=END_ID,
        skip_ids=SKIP_IDS,
    )
