import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

# ==========================
# 配置区：PyCharm 只改这里
# ==========================
SPACY_MODEL = "en_core_web_sm"

INPUT_PATH = "simplified_train.json"   # list[{Id, Question, ...}]
OUTPUT_PATH = "227dep_relation_train_all.json"  # list[{Id, relation, syntax_tree}]

BATCH_SIZE = 64

# 处理范围控制
MAX_N = None            # 只处理前 MAX_N 条；None=全量
QUESTION_NUM = None     # 只处理最后 QUESTION_NUM 条；None=不用
START_INDEX = 6000      # 手动指定起始 index（优先级最高）
END_INDEX = None        # 手动指定结束 index（不含）

# ==========================
# 依存抽取规则参数
# ==========================
# 只把这些介词算进谓词短语（避免 join after / win during 这种时间介词污染）
ALLOWED_PREPS = {"to", "of", "in", "for", "from", "at", "on", "with", "as"}

# 这些“连系/准连系动词”可扩展为：become + attr(+prep) -> become director of
PREDICATIVE_VERBS = {"become", "remain", "stay", "seem", "appear"}

MAX_REL_TOKENS = 4  # relation 最多 1~4 token


def load_json_list(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else []


def compute_range(total: int) -> Tuple[int, int]:
    if MAX_N is None:
        max_n = total
    else:
        max_n = min(MAX_N, total)

    if END_INDEX is not None:
        end_index = max(0, min(END_INDEX, total))
    else:
        end_index = max_n

    if START_INDEX is not None:
        start_index = max(0, min(START_INDEX, total))
    elif QUESTION_NUM is not None:
        start_index = max(0, end_index - QUESTION_NUM)
    else:
        start_index = 0

    if start_index >= end_index:
        raise RuntimeError(f"无效区间：start_index={start_index}, end_index={end_index}, total={total}")
    return start_index, end_index


def token_key(tok) -> str:
    """
    基础谓词 token：
    - VERB 用 lemma（join / win / belong）
    - ADJ 默认用 lower_（避免 born -> bear 这种 lemma 不符合直觉）
    - NOUN 用 lemma（member / director）
    """
    low = tok.lower_
    if low == "born":  # 特判：born 的 lemma=bear，不想要
        return "born"
    if tok.pos_ == "VERB":
        # return tok.lemma_.lower()
        return tok.lower() #取消词形还原
    if tok.pos_ in {"NOUN", "PROPN"}:
        return tok.lemma_.lower()
    return low


def compound_phrase(tok) -> str:
    """
    对名词谓词补全 compound（prime minister / vice president）。
    spaCy: compound 依存一般挂在名词 head 上。
    """
    parts = [tok]
    for c in tok.children:
        if c.dep_ == "compound":
            parts.append(c)
    parts = sorted(set(parts), key=lambda x: x.i)
    # 对 compound 仍用 surface 更自然（prime / vice）
    phrase = " ".join([p.lower_ if p.dep_ == "compound" else token_key(p) for p in parts])
    return phrase.strip()


def pick_prep(head_tok) -> Optional[str]:
    """
    取 head_tok 的介词依存（prep），仅允许 ALLOWED_PREPS 里的。
    """
    for c in head_tok.children:
        if c.dep_ == "prep" and c.lower_ in ALLOWED_PREPS:
            return c.lower_
    return None


def find_root(doc):
    for t in doc:
        if t.dep_ == "ROOT":
            return t
    return doc[0] if len(doc) else None


def dep_tree_conll(doc) -> str:
    """
    输出 CoNLL 风格的依存树（字符串）。
    列：i, text, lemma, UPOS, dep, head_i, head_text
    head_i 用 0 表示 ROOT。
    """
    lines = []
    for t in doc:
        i = t.i + 1
        head_i = 0 if t.dep_ == "ROOT" else (t.head.i + 1)
        head_text = "ROOT" if t.dep_ == "ROOT" else t.head.text
        lines.append(
            f"{i}\t{t.text}\t{t.lemma_}\t{t.pos_}\t{t.dep_}\t{head_i}\t{head_text}"
        )
    return "\n".join(lines)


def extract_relation(doc) -> str:
    """
    规则化抽取中心谓词/谓词短语：
    1) 取 ROOT
    2) VERB:
       - relation = lemma(ROOT)
       - 若有 particle(prt)：加上（pass away / take over）
       - 若有 allowed prep：加上（belong to）
       - 若是 PREDICATIVE_VERBS：可扩展 attr + prep -> become director of
    3) NOUN/ADJ（系动词句常见）：relation = (compound+)ROOT + allowed prep -> member of / born in
    """
    root = find_root(doc)
    if root is None:
        return ""

    toks: List[str] = []

    # ---------- case A: VERB ----------
    if root.pos_ == "VERB":
        base = token_key(root)
        toks.append(base)

        # particle: prt
        prts = [c for c in root.children if c.dep_ == "prt"]
        if prts:
            toks.append(prts[0].lower_)

        # verb + allowed prep（belong to / refer to）
        prep = pick_prep(root)
        if prep:
            toks.append(prep)

        # predicative verb 扩展：become + attr(+prep)
        if root.lemma_.lower() in PREDICATIVE_VERBS:
            # attr/oprd/acomp 常见于系表补语；不同版本模型可能略有差异
            candidates = [c for c in root.children if c.dep_ in {"attr", "oprd", "acomp"}]
            if not candidates:
                # 兜底：有时补语会被标成 dobj/obj
                candidates = [c for c in root.children if c.dep_ in {"dobj", "obj"}]
            if candidates:
                comp = candidates[0]
                if comp.pos_ in {"NOUN", "PROPN", "ADJ"}:
                    toks.append(compound_phrase(comp) if comp.pos_ != "ADJ" else comp.lower_)
                    comp_prep = pick_prep(comp)
                    if comp_prep:
                        toks.append(comp_prep)

    # ---------- case B: NOUN/ADJ/PROPN ----------
    else:
        # 名词谓词：补 compound（prime minister）
        if root.pos_ in {"NOUN", "PROPN"}:
            toks.append(compound_phrase(root))
        else:
            toks.append(token_key(root))

        prep = pick_prep(root)
        if prep:
            toks.append(prep)

    # 截断到 1~4 token
    toks = [t for t in toks if t]
    return " ".join(toks[:MAX_REL_TOKENS]).strip()


def normalize_existing(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    if "Id" not in item or "relation" not in item or "syntax_tree" not in item:
        return None
    try:
        _id = int(item["Id"])
    except Exception:
        return None
    rel = str(item["relation"]).strip()
    tree = str(item["syntax_tree"]).strip()
    if not rel or not tree:
        return None
    return {"Id": _id, "relation": rel, "syntax_tree": tree}


def main():
    try:
        import spacy
    except ImportError:
        raise RuntimeError("未安装 spaCy：请先运行 pip install -U spacy")

    try:
        nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat"])
    except Exception:
        raise RuntimeError(f"未安装模型 {SPACY_MODEL}：请运行 python -m spacy download {SPACY_MODEL}")

    data = load_json_list(INPUT_PATH)
    if not data:
        raise RuntimeError(f"输入文件为空或不是 JSON list：{INPUT_PATH}")

    total = len(data)
    start_index, end_index = compute_range(total)

    # 断点续跑
    existing_by_id: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            existing = load_json_list(OUTPUT_PATH)
            for it in existing:
                norm = normalize_existing(it) if isinstance(it, dict) else None
                if norm:
                    existing_by_id[norm["Id"]] = norm
        except Exception:
            pass

    print(f"[config] total={total}, range=[{start_index},{end_index}), batch_size={BATCH_SIZE}, already_saved={len(existing_by_id)}")

    t0 = time.time()

    # 逐批处理（用 nlp.pipe 更快）
    idx = start_index
    while idx < end_index:
        chunk = data[idx: min(idx + BATCH_SIZE, end_index)]

        # 收集待处理
        ids: List[int] = []
        questions: List[str] = []
        for x in chunk:
            if not isinstance(x, dict):
                continue
            if "Id" not in x or "Question" not in x:
                continue
            _id = int(x["Id"])
            if _id in existing_by_id:
                continue
            q = str(x["Question"])
            ids.append(_id)
            questions.append(q)

        if not ids:
            idx += BATCH_SIZE
            continue

        print(f"[batch] idx {idx}~{min(idx+BATCH_SIZE-1, end_index-1)} | new={len(ids)} | ids={ids[:5]}{'...' if len(ids)>5 else ''}")

        # 解析并抽取
        for _id, doc in zip(ids, nlp.pipe(questions, batch_size=32)):
            rel = extract_relation(doc)
            tree = dep_tree_conll(doc)
            existing_by_id[_id] = {"Id": _id, "relation": rel, "syntax_tree": tree}

        # 写回
        merged = [existing_by_id[k] for k in sorted(existing_by_id.keys())]
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        print(f"[saved] total_saved={len(merged)} | elapsed={elapsed:.1f}s -> {OUTPUT_PATH}")

        idx += BATCH_SIZE

    print(f"Done. Output -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()