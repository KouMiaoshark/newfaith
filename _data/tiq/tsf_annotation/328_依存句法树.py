import json
import time
from typing import Any, Dict, List, Optional, Tuple

# ==========================
# 配置区：只改这里
# ==========================
SPACY_MODEL = "en_core_web_sm"

INPUT_PATH = "simplified_dev.json"               # 输入：list[{"Id","Question",...}]
OUTPUT_PATH = "328dep_relation_dev_all.json"     # 输出：list[{"Id","question","relation","syntax_tree"}]

BATCH_SIZE = 64

# 处理范围控制
MAX_N = None
QUESTION_NUM = None
START_INDEX = 0
END_INDEX = None

# ==========================
# relation 抽取规则参数
# ==========================
# 只保留关系型介词，不保留时间型介词
REL_PREPS = {"of", "in", "at", "to", "for", "from", "with", "as", "on"}
TEMPORAL_PREPS = {"during", "before", "after", "when", "while", "since", "until"}

# 可扩展为谓词补语结构的动词
PREDICATIVE_VERBS = {"become", "remain", "stay", "seem", "appear"}

# 疑问词：不允许进入 relation
QUESTION_WORDS = {
    "who", "what", "which", "whom", "whose",
    "when", "where", "why", "how"
}

# 这些词在依存里常被打成 amod，但语义上其实是职位/头衔/固定称谓的一部分
TITLELIKE_AMODS = {
    "chief", "prime", "vice", "deputy", "assistant", "associate",
    "acting", "interim", "executive", "general", "royal", "senior",
    "junior", "lead", "head"
}

# 轻动词结构中的名词头：held the position of ...
LIGHT_VERB_OBJECT_HEADS = {
    "position", "role", "office", "post", "title", "job"
}

MAX_REL_TOKENS = 6


# =========================================================
# 基础 IO
# =========================================================
def load_json_list(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else []


def save_json_list(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
        raise RuntimeError(
            f"无效区间：start_index={start_index}, end_index={end_index}, total={total}"
        )
    return start_index, end_index


# =========================================================
# spaCy 辅助函数
# =========================================================
def is_be_token(tok) -> bool:
    if tok is None:
        return False
    return tok.lemma_.lower() == "be" or tok.lower_ in {
        "am", "is", "are", "was", "were", "be", "been", "being"
    }


def is_question_word_token(tok) -> bool:
    return tok is not None and tok.lower_ in QUESTION_WORDS


def find_root(doc):
    for t in doc:
        if t.dep_ == "ROOT":
            return t
    return doc[0] if len(doc) else None


def safe_lower(tok) -> str:
    """
    尽量保留 surface lower，而不是强行 lemma。
    避免 born -> bear 这种问题。
    """
    low = tok.lower_
    if low == "born":
        return "born"
    return low


def token_key(tok) -> str:
    """
    relation token 默认保留表面形式的小写。
    """
    return safe_lower(tok)


def sorted_unique_tokens(tokens):
    uniq = {}
    for t in tokens:
        uniq[t.i] = t
    return [uniq[i] for i in sorted(uniq.keys())]


def clean_relation_tokens(tokens: List[str]) -> List[str]:
    """
    最终 relation token 清洗：
    1. 去空
    2. 去疑问词
    3. 去连续重复
    """
    cleaned = []
    for tok in tokens:
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in QUESTION_WORDS:
            continue
        if cleaned and cleaned[-1] == tok:
            continue
        cleaned.append(tok)
    return cleaned


def find_first_child(head, dep_names: set):
    candidates = [c for c in head.children if c.dep_ in dep_names]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.i)[0]


def find_children(head, dep_names: set) -> List[Any]:
    return sorted([c for c in head.children if c.dep_ in dep_names], key=lambda x: x.i)


def choose_relation_prep(head) -> Optional[str]:
    """
    只选关系型介词，不选时间型介词。
    """
    preps = find_children(head, {"prep"})
    for p in preps:
        low = p.lower_
        if low in TEMPORAL_PREPS:
            continue
        if low in REL_PREPS:
            return low
    return None


def choose_relation_prep_node(head):
    """
    返回第一个关系型介词节点本身，便于继续取其 pobj。
    """
    preps = find_children(head, {"prep"})
    for p in preps:
        low = p.lower_
        if low in TEMPORAL_PREPS:
            continue
        if low in REL_PREPS:
            return p
    return None


def choose_particle(head) -> Optional[str]:
    prts = find_children(head, {"prt"})
    if prts:
        return prts[0].lower_
    return None


def choose_aux_for_lexical_verb(head):
    """
    对真正动词，恢复其左侧问句助动词：did join / was representing。
    只取最常见的 do/be 系辅助词，避免把情态等乱带入。
    """
    auxes = [
        c for c in find_children(head, {"aux", "auxpass"})
        if c.lower_ in {"do", "does", "did", "is", "are", "was", "were", "be", "been", "being"}
    ]
    if not auxes:
        return None
    return auxes[0]


def normalize_temporal_root_predicate(tok):
    """
    时间前置场景下，若误选到了 AUX（如 did），尝试沿 head 恢复到真正动词（如 join）。
    """
    if tok is None:
        return None
    if tok.pos_ == "AUX" and tok.head is not None and tok.head.pos_ == "VERB" and tok.head.i > tok.i:
        return tok.head
    return tok


def get_left_modifiers_for_nominal(head) -> List[Any]:
    """
    给名词性谓词补全左侧核心修饰：
    - 保留 compound（如 home venue / prime minister）
    - 对极少数 title-like amod 也保留（如 chief minister / prime minister）
    - 不普遍保留 amod，避免 major city 这种答案类型描述混入 relation
    """
    mods = []
    for c in head.children:
        if c.i >= head.i:
            continue
        if c.dep_ == "compound":
            mods.append(c)
        elif c.dep_ == "amod" and c.lower_ in TITLELIKE_AMODS:
            mods.append(c)
    return sorted(mods, key=lambda x: x.i)


def nominal_phrase_tokens(head) -> List[str]:
    toks = get_left_modifiers_for_nominal(head) + [head]
    toks = sorted_unique_tokens(toks)
    words = [token_key(t) for t in toks if t.lower_ not in QUESTION_WORDS]
    return words


def nominal_phrase(head) -> str:
    return " ".join(nominal_phrase_tokens(head)).strip()


def relation_pobj_tokens(prep_node) -> List[str]:
    """
    给定关系型介词节点，补出介词宾语短语。
    只补一层，避免把地点/时间等后续状语都卷进 relation。
    """
    if prep_node is None:
        return []

    pobj = find_first_child(prep_node, {"pobj"})
    if pobj is None:
        return []

    if is_question_word_token(pobj):
        return []

    toks = [prep_node.lower_]
    if pobj.pos_ in {"NOUN", "PROPN"}:
        toks.extend(nominal_phrase_tokens(pobj))
    else:
        toks.append(token_key(pobj))
    return toks


def nominal_relation_tokens(head) -> List[str]:
    """
    名词性关系中心：
    - head
    - 若自带关系型介词，则补成 head + prep + pobj
    例：
    - minister -> chief minister
    - head -> head of government
    """
    toks = nominal_phrase_tokens(head)
    prep_node = choose_relation_prep_node(head)
    if prep_node is not None:
        toks.extend(relation_pobj_tokens(prep_node))
    return toks


def choose_acl_action_from_attr_for_be(root):
    """
    针对 be-root 的另一类错误分析：
    ROOT=be，attr 被打成人名/实体，而真正关系动词挂在 attr 下面的 acl/relcl。
    例：
      What canton was Lugano located in ...
      ROOT=was, attr=Lugano, Lugano --acl--> located
      => was located in
    """
    attrs = [a for a in find_children(root, {"attr"}) if not is_question_word_token(a)]
    if not attrs:
        return None

    for attr in attrs:
        if attr.pos_ not in {"PROPN", "NOUN"}:
            continue
        verbal_children = [
            c for c in find_children(attr, {"acl", "relcl"})
            if c.pos_ == "VERB" and not is_be_token(c) and not is_question_word_token(c)
        ]
        if verbal_children:
            return verbal_children[0]
    return None


def choose_advcl_action_for_be(root):
    """
    针对 be-root 的错误分析场景：
    若 attr 像人名/实体名，而右边 advcl 是真正动作，则优先动作。
    例：
      what team was Carla Overbeck representing ...
      ROOT=was, attr=Overbeck, advcl=representing
      => was representing
    """
    attrs = find_children(root, {"attr"})
    if not attrs:
        return None

    # attr 基本像专名/人名，说明它不太可能是真正 relation 中心
    if not all(a.pos_ == "PROPN" for a in attrs if not is_question_word_token(a)):
        return None

    candidates = [
        c for c in find_children(root, {"advcl", "ccomp", "xcomp"})
        if c.pos_ == "VERB" and not is_be_token(c) and not is_question_word_token(c)
    ]
    if not candidates:
        return None
    return candidates[0]


def choose_predicative_complement(head):
    """
    适用于：
    1) be-root: was governor / was coaching / was wife
    2) become/remain-root: became director / remained member

    修正要点：
    - 对 be-root，优先 ccomp/xcomp（如 was coaching）
    - 若 attr 像人名而 advcl 才是真动作，则优先 advcl（如 was representing）
    - 若真正动作挂在 attr 的 acl/relcl 下，则优先该动作（如 was located in）
    - 再看 attr/acomp/oprd
    - 最后才看 obj
    """
    if is_be_token(head):
        advcl_action = choose_advcl_action_for_be(head)
        if advcl_action is not None:
            return advcl_action

        acl_action = choose_acl_action_from_attr_for_be(head)
        if acl_action is not None:
            return acl_action

        priority_dep_sets = [
            {"ccomp", "xcomp"},
            {"attr", "acomp", "oprd"},
            {"dobj", "obj"},
        ]
    else:
        priority_dep_sets = [
            {"attr", "acomp", "oprd"},
            {"ccomp", "xcomp"},
            {"dobj", "obj"},
        ]

    for dep_set in priority_dep_sets:
        candidates = find_children(head, dep_set)
        if not candidates:
            continue

        non_q = [c for c in candidates if not is_question_word_token(c)]
        if not non_q:
            return candidates[0]

        # 对 ccomp/xcomp：优先真正动词
        verbal = [c for c in non_q if c.pos_ in {"VERB", "AUX"}]
        if verbal and dep_set == {"ccomp", "xcomp"}:
            return verbal[0]

        preferred = [c for c in non_q if c.pos_ in {"NOUN", "PROPN", "ADJ", "VERB", "AUX"}]
        if preferred:
            return preferred[0]

        return non_q[0]

    return None


def choose_nonwh_subject_for_be(root):
    """
    be-root 的回退：
    优先找非疑问词主语中的名词性中心。
    例：
        who was his wife? -> wife
    """
    subjects = find_children(root, {"nsubj", "nsubjpass"})
    if not subjects:
        return None

    non_q = [s for s in subjects if not is_question_word_token(s)]
    if not non_q:
        return None

    preferred = [s for s in non_q if s.pos_ in {"NOUN", "PROPN"}]
    if preferred:
        return preferred[0]

    return non_q[0]


def find_nominal_relation_from_subject_for_be(root):
    """
    当 ROOT 是 be 动词，但直接补语没法给出有效 relation 时，尝试从主语恢复关系。

    两类：
    1) be -> nsubj(head)，且 head 自身带关系型介词
       例如：
         was --nsubj--> head
         head --prep--> of
         of --pobj--> government
       => head of government

    2) be -> nsubj(X) -> prep(of/...) -> pobj(Y)
       例如更深一层的名词链
       => Y (+ prep)
    """
    subj = choose_nonwh_subject_for_be(root)
    if subj is None:
        return None, []

    # 先看主语本身是不是关系中心：head of government
    subj_prep = choose_relation_prep_node(subj)
    if subj.pos_ in {"NOUN", "PROPN"} and subj_prep is not None:
        return subj, relation_pobj_tokens(subj_prep)

    # 再看 subj -> prep -> pobj 是否能给出更深层中心
    for prep in find_children(subj, {"prep"}):
        prep_low = prep.lower_
        if prep_low in TEMPORAL_PREPS or prep_low not in REL_PREPS:
            continue

        pobj = find_first_child(prep, {"pobj"})
        if pobj is None:
            continue
        if pobj.pos_ not in {"NOUN", "PROPN", "ADJ"}:
            continue
        if is_question_word_token(pobj):
            continue

        trailing_tokens = []
        trailing_prep = choose_relation_prep_node(pobj)
        if trailing_prep is not None:
            trailing_tokens = relation_pobj_tokens(trailing_prep)

        return pobj, trailing_tokens

    return subj, []


def choose_lexical_verb_for_do_aux(root):
    """
    处理 do/does/did 作为 ROOT 的情况。
    """
    if root.lower_ not in {"do", "does", "did"}:
        return None

    for dep_set in [{"xcomp", "ccomp"}, {"dobj", "obj"}]:
        candidates = find_children(root, dep_set)
        if not candidates:
            continue

        candidates = [c for c in candidates if not is_question_word_token(c)]
        if not candidates:
            continue

        verbal = [c for c in candidates if c.pos_ in {"VERB", "AUX"}]
        if verbal:
            return verbal[0]

        return candidates[0]

    return None


def choose_light_verb_relation(root) -> Optional[List[str]]:
    """
    轻动词结构：
      held the position of Minority Whip
      -> held the position of

    只保留到抽象名词头 + 介词，不把具体职位名卷入 relation，
    避免把 answer type/具体答案污染 relation。
    """
    if root.pos_ != "VERB":
        return None

    dobj = find_first_child(root, {"dobj", "obj"})
    if dobj is None or dobj.pos_ not in {"NOUN", "PROPN"}:
        return None
    if dobj.lemma_.lower() not in LIGHT_VERB_OBJECT_HEADS and dobj.lower_ not in LIGHT_VERB_OBJECT_HEADS:
        return None

    prep_node = choose_relation_prep_node(dobj)
    if prep_node is None or prep_node.lower_ != "of":
        return None

    toks = [token_key(root)]
    dets = [c for c in find_children(dobj, {"det"}) if c.lower_ in {"the", "a", "an"}]
    if dets:
        toks.append(token_key(dets[0]))
    toks.extend(nominal_phrase_tokens(dobj))
    toks.append("of")
    return toks


def choose_main_predicate_for_temporal_marker_root(doc, root):
    """
    处理 ROOT 被错误打成 after/before/during/when 等时间引导词的情况。

    关键修正：
    不再排除整个 pcomp.subtree，
    只跳过时间短语内部最明显的几个 token：
    - root 自己
    - pcomp 自己（如 born / coaching）
    - pcomp 的 aux / auxpass / mark

    然后在剩余句子里继续找真正主句谓词。
    """
    skip_ids = {root.i}

    pcomp = find_first_child(root, {"pcomp"})
    if pcomp is not None:
        skip_ids.add(pcomp.i)
        for c in pcomp.children:
            if c.dep_ in {"aux", "auxpass", "mark"}:
                skip_ids.add(c.i)

    candidates = []
    for tok in doc:
        if tok.i in skip_ids:
            continue
        if tok.i <= root.i:
            continue
        if is_question_word_token(tok):
            continue
        if tok.pos_ not in {"VERB", "AUX"}:
            continue
        candidates.append(tok)

    if not candidates:
        return None

    # 1) 优先 非be 的真正实义动词，且尽量有关系介词
    preferred = [
        tok for tok in candidates
        if tok.pos_ == "VERB" and (not is_be_token(tok)) and choose_relation_prep(tok) is not None
    ]
    if preferred:
        return preferred[0]

    # 2) 再优先 非be 的真正实义动词（避免 did 抢到 join 前面）
    preferred = [tok for tok in candidates if tok.pos_ == "VERB" and not is_be_token(tok)]
    if preferred:
        return preferred[0]

    # 3) 再退到其他非be动词/AUX
    preferred = [tok for tok in candidates if not is_be_token(tok)]
    if preferred:
        return preferred[0]

    # 4) 最后才退化到 be
    return candidates[0]


# =========================================================
# 依存树输出
# =========================================================
def dep_tree_arrows(doc) -> str:
    """
    箭头风格依存树，便于人工查看
    """
    lines = []
    root = find_root(doc)
    if root is None:
        return ""

    lines.append(f"ROOT --> {root.text}({root.i + 1})")
    for tok in doc:
        if tok.dep_ == "ROOT":
            continue
        head = tok.head
        lines.append(
            f"{head.text}({head.i + 1}) --{tok.dep_}--> {tok.text}({tok.i + 1})"
        )
    return "\n".join(lines)


# =========================================================
# relation 抽取核心
# =========================================================
def extract_relation(doc) -> str:
    """
    依存规则版 relation 抽取：

    A. ROOT 是时间引导词 after/before/during/when
       -> 跳过时间根，去主句里找真正谓词
       例：
       - After being born ..., which championship ... race in?
         => race in

    B. ROOT 是名词/形容词，且带 cop(be)
       -> be + 表语中心 (+ prep/+pobj)

    C. ROOT 是 be 动词
       -> be + 补语中心 (+ prep/+pobj)
       若直接补语无效，则尝试：
       -> be + 主语内部名词关系中心

    D. ROOT 是 predicative verb
       -> verb + complement (+ prep/+pobj)

    E. ROOT 是 do/does/did 助动词
       -> do/does/did + lexical verb (+ prep)

    F. ROOT 是普通实义动词
       -> verb (+ particle) (+ prep)
    """
    root = find_root(doc)
    if root is None:
        return ""

    rel_tokens: List[str] = []

    # -----------------------------------------------------
    # Case 0: ROOT 被错误打成时间引导词
    # -----------------------------------------------------
    if root.lower_ in TEMPORAL_PREPS:
        main_pred = choose_main_predicate_for_temporal_marker_root(doc, root)
        main_pred = normalize_temporal_root_predicate(main_pred)
        if main_pred is not None:
            aux = choose_aux_for_lexical_verb(main_pred)
            if aux is not None:
                rel_tokens.append(token_key(aux))

            rel_tokens.append(token_key(main_pred))

            prt = choose_particle(main_pred)
            if prt:
                rel_tokens.append(prt)

            prep = choose_relation_prep(main_pred)
            if prep:
                rel_tokens.append(prep)

        rel_tokens = clean_relation_tokens(rel_tokens)
        return " ".join(rel_tokens[:MAX_REL_TOKENS]).strip()

    # -----------------------------------------------------
    # Case 1: ROOT 是名词/形容词，且有 cop(be)
    # -----------------------------------------------------
    cop_child = None
    for c in root.children:
        if c.dep_ == "cop" and is_be_token(c):
            cop_child = c
            break

    if cop_child is not None and root.pos_ in {"NOUN", "PROPN", "ADJ"}:
        rel_tokens.append(token_key(cop_child))

        if root.pos_ in {"NOUN", "PROPN"}:
            rel_tokens.extend(nominal_relation_tokens(root))
        else:
            rel_tokens.append(token_key(root))
            prep = choose_relation_prep(root)
            if prep:
                rel_tokens.append(prep)

        rel_tokens = clean_relation_tokens(rel_tokens)
        return " ".join(rel_tokens[:MAX_REL_TOKENS]).strip()

    # -----------------------------------------------------
    # Case 2: ROOT 是 be 动词
    # -----------------------------------------------------
    if is_be_token(root):
        rel_tokens.append(token_key(root))

        comp = choose_predicative_complement(root)

        if comp is not None and not is_question_word_token(comp):
            if comp.pos_ in {"VERB", "AUX"}:
                rel_tokens.append(token_key(comp))
                prep = choose_relation_prep(comp)
                if prep:
                    rel_tokens.append(prep)
            elif comp.pos_ in {"NOUN", "PROPN"}:
                rel_tokens.extend(nominal_relation_tokens(comp))
            else:
                rel_tokens.append(token_key(comp))
                prep = choose_relation_prep(comp)
                if prep:
                    rel_tokens.append(prep)
        else:
            subj_nominal, trailing_tokens = find_nominal_relation_from_subject_for_be(root)
            if subj_nominal is not None:
                if subj_nominal.pos_ in {"NOUN", "PROPN"}:
                    rel_tokens.extend(nominal_phrase_tokens(subj_nominal))
                else:
                    rel_tokens.append(token_key(subj_nominal))
                rel_tokens.extend(trailing_tokens)

        rel_tokens = clean_relation_tokens(rel_tokens)
        return " ".join(rel_tokens[:MAX_REL_TOKENS]).strip()

    # -----------------------------------------------------
    # Case 3: ROOT 是 predicative verb
    # -----------------------------------------------------
    if root.lemma_.lower() in PREDICATIVE_VERBS or root.lower_ in PREDICATIVE_VERBS:
        rel_tokens.append(token_key(root))

        comp = choose_predicative_complement(root)
        if comp is not None and not is_question_word_token(comp):
            if comp.pos_ in {"NOUN", "PROPN"}:
                rel_tokens.extend(nominal_relation_tokens(comp))
            else:
                rel_tokens.append(token_key(comp))
                prep = choose_relation_prep(comp)
                if prep:
                    rel_tokens.append(prep)

        rel_tokens = clean_relation_tokens(rel_tokens)
        return " ".join(rel_tokens[:MAX_REL_TOKENS]).strip()

    # -----------------------------------------------------
    # Case 4: ROOT 是 do/does/did 助动词
    # -----------------------------------------------------
    if root.lower_ in {"do", "does", "did"}:
        rel_tokens.append(token_key(root))

        lexical = choose_lexical_verb_for_do_aux(root)
        if lexical is not None:
            rel_tokens.append(token_key(lexical))

            prep = choose_relation_prep(lexical)
            if prep:
                rel_tokens.append(prep)

        rel_tokens = clean_relation_tokens(rel_tokens)
        return " ".join(rel_tokens[:MAX_REL_TOKENS]).strip()

    # -----------------------------------------------------
    # Case 5: ROOT 是普通实义动词
    # -----------------------------------------------------
    if root.pos_ in {"VERB", "AUX"}:
        light_relation = choose_light_verb_relation(root)
        if light_relation is not None:
            rel_tokens.extend(light_relation)
        else:
            rel_tokens.append(token_key(root))

            prt = choose_particle(root)
            if prt:
                rel_tokens.append(prt)

            prep = choose_relation_prep(root)
            if prep:
                rel_tokens.append(prep)

        rel_tokens = clean_relation_tokens(rel_tokens)
        return " ".join(rel_tokens[:MAX_REL_TOKENS]).strip()

    # -----------------------------------------------------
    # Case 6: 兜底
    # -----------------------------------------------------
    if root.pos_ in {"NOUN", "PROPN"}:
        rel_tokens.extend(nominal_relation_tokens(root))
    else:
        rel_tokens.append(token_key(root))

    rel_tokens = clean_relation_tokens(rel_tokens)
    return " ".join(rel_tokens[:MAX_REL_TOKENS]).strip()


# =========================================================
# 主流程
# =========================================================
def main():
    try:
        import spacy
    except ImportError:
        raise RuntimeError("未安装 spaCy，请先运行：pip install -U spacy")

    try:
        nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat"])
    except Exception:
        raise RuntimeError(
            f"未安装模型 {SPACY_MODEL}，请先运行：python -m spacy download {SPACY_MODEL}"
        )

    data = load_json_list(INPUT_PATH)
    if not data:
        raise RuntimeError(f"输入文件为空或不是 JSON list：{INPUT_PATH}")

    total = len(data)
    start_index, end_index = compute_range(total)

    print(
        f"[config] total={total}, range=[{start_index},{end_index}), "
        f"batch_size={BATCH_SIZE}, output={OUTPUT_PATH}"
    )

    # 每次都从空结果开始，最终覆盖旧文件
    results: List[Dict[str, Any]] = []

    t0 = time.time()
    idx = start_index

    while idx < end_index:
        chunk = data[idx:min(idx + BATCH_SIZE, end_index)]

        items_to_process: List[Tuple[int, str]] = []
        for item in chunk:
            if not isinstance(item, dict):
                continue
            if "Id" not in item or "Question" not in item:
                continue

            _id = int(item["Id"])
            question = str(item["Question"]).strip()
            if not question:
                continue

            items_to_process.append((_id, question))

        if not items_to_process:
            idx += BATCH_SIZE
            continue

        ids = [x[0] for x in items_to_process]
        questions = [x[1] for x in items_to_process]

        print(
            f"[batch] idx {idx}~{min(idx + BATCH_SIZE - 1, end_index - 1)} "
            f"| count={len(ids)} | ids={ids[:5]}{'...' if len(ids) > 5 else ''}"
        )

        for _id, question, doc in zip(ids, questions, nlp.pipe(questions, batch_size=32)):
            relation = extract_relation(doc)
            syntax_tree = dep_tree_arrows(doc)
            results.append({
                "Id": _id,
                "question": question,
                "relation": relation,
                "syntax_tree": syntax_tree,
            })

        elapsed = time.time() - t0
        print(f"[progress] processed={len(results)} | elapsed={elapsed:.1f}s")

        idx += BATCH_SIZE

    # 按 Id 排序后统一写出，覆盖旧文件
    results = sorted(results, key=lambda x: x["Id"])
    save_json_list(OUTPUT_PATH, results)

    print(f"Done. Output overwritten -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()