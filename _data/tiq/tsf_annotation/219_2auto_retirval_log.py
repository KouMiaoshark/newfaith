import json
import time
import random
import os
import re
import requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_PAGE = "https://www.wikidata.org/wiki/{}"

# ========= 只需要改这里 =========
input_path = r"历史提示词处理/simplify_rela_entity_answer_dev_solution9_implicit_6001-6301_dev.json"
output_path = None  # None = 自动生成：<input>_wikidata_check.json
# =================================

# ====== 检索参数 ======
SEARCH_LANGUAGE = "en"
SEARCH_LIMIT = 5
REQUEST_TIMEOUT = 20
SLEEP_RANGE = (0.15, 0.35)
MAX_RETRY = 5

HIT_AS_LIST = True
STRICT_BY_ANSWER_QID = True  # True: 页面必须包含 Answer 的 QID 才算命中；False: 只查 label(类似 Ctrl+F)

# ====== 日志参数（新增） ======
LOG_ENABLE = True
LOG_EVERY_Q = 10            # 每处理多少条问题打印一次总进度（1=每条都打印）
LOG_ENTITY_PROGRESS = True  # 是否打印 origin/entity 内部实体进度（会更详细但不至于刷屏）
LOG_EVERY_ENTITY = 5        # 每检查多少个实体名打印一次（仅在 LOG_ENTITY_PROGRESS=True 时生效）
LOG_RETRY = True            # 请求重试时是否打印
LOG_TO_FILE = False         # True: 同时写入日志文件
LOG_FILE_PATH = "wikidata_check.log"


def log(msg: str) -> None:
    if not LOG_ENABLE:
        return
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_TO_FILE:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _sleep():
    time.sleep(random.uniform(*SLEEP_RANGE))


def _request_with_retry(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)

            # 429/5xx：退避重试
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if LOG_RETRY:
                    log(f"HTTP {resp.status_code} retry {attempt}/{MAX_RETRY} for {url}")
                wait = min(2 ** attempt, 20) + random.uniform(0, 0.5)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        except requests.RequestException as e:
            last_exc = e
            if attempt == MAX_RETRY:
                break
            if LOG_RETRY:
                log(f"Request error retry {attempt}/{MAX_RETRY} for {url}: {repr(e)}")
            wait = min(2 ** attempt, 20) + random.uniform(0, 0.5)
            time.sleep(wait)

    raise RuntimeError(f"Request failed after retries: {url}. Last error: {repr(last_exc)}")


def wikidata_search_qid(session: requests.Session, query: str, language: str, limit: int) -> Optional[str]:
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": language,
        "format": "json",
        "limit": limit,
    }
    resp = _request_with_retry(session, "GET", WIKIDATA_API, params=params)
    data = resp.json()
    results = data.get("search", [])
    if not results:
        return None
    return results[0].get("id")  # e.g., "Q18810082"


def fetch_entity_page_html(session: requests.Session, qid: str) -> str:
    url = WIKIDATA_ENTITY_PAGE.format(qid)
    resp = _request_with_retry(session, "GET", url)
    return resp.text


def extract_answer_labels_and_qids(item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    labels: List[str] = []
    qids: List[str] = []
    for ans in item.get("Answer", []) or []:
        lab = ans.get("WikidataLabel")
        aqid = ans.get("WikidataQid")
        if lab:
            labels.append(str(lab).strip())
        if aqid:
            qids.append(str(aqid).strip())

    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return uniq(labels), uniq(qids)


def page_contains_by_label(html: str, answer_labels: List[str]) -> List[str]:
    h = html.lower()
    hit = []
    for lab in answer_labels:
        if lab and lab.lower() in h:
            hit.append(lab)
    return hit


def page_contains_by_answer_qid(html: str, answer_qids: List[str]) -> List[str]:
    hit = []
    for q in answer_qids:
        if not q:
            continue
        if (f'/wiki/{q}' in html) or (f'data-entityid="{q}"' in html) or re.search(rf'\b{re.escape(q)}\b', html):
            hit.append(q)
    return hit


def check_entity_list_for_answer(
    session: requests.Session,
    question_id: Any,
    list_name: str,  # "origin_entity" or "entity"
    entity_names: List[str],
    answer_labels: List[str],
    answer_qids: List[str],
    qid_cache: Dict[str, Optional[str]],
    html_cache: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    hits: List[str] = []
    detail: Dict[str, Any] = {}

    total = len(entity_names)
    checked = 0

    for idx, name in enumerate(entity_names, start=1):
        name = str(name).strip()
        if not name:
            continue

        # 日志：实体列表内进度
        if LOG_ENTITY_PROGRESS and (idx == 1 or idx % LOG_EVERY_ENTITY == 0 or idx == total):
            log(f"Id={question_id} {list_name}: checking entity {idx}/{total} -> '{name}'")

        # 1) 搜索得到实体 QID（缓存）
        if name in qid_cache:
            qid = qid_cache[name]
        else:
            _sleep()
            qid = wikidata_search_qid(session, name, SEARCH_LANGUAGE, SEARCH_LIMIT)
            qid_cache[name] = qid

        detail[name] = {
            "searched_qid": qid,   # 关键：你关心的“是否真的拿到了 QID”
            "hit_by_label": [],
            "hit_by_answer_qid": [],
            "status": "searched",
        }

        if not qid:
            detail[name]["status"] = "no_qid"
            checked += 1
            continue

        # 2) 打开 QID 页面 HTML（缓存）
        if qid in html_cache:
            html = html_cache[qid]
        else:
            _sleep()
            html = fetch_entity_page_html(session, qid)
            html_cache[qid] = html

        # 3) 检索答案
        hit_labels = page_contains_by_label(html, answer_labels)
        hit_aqids = page_contains_by_answer_qid(html, answer_qids)

        detail[name]["hit_by_label"] = hit_labels
        detail[name]["hit_by_answer_qid"] = hit_aqids

        if STRICT_BY_ANSWER_QID:
            ok = len(hit_aqids) > 0
        else:
            ok = len(hit_labels) > 0

        detail[name]["status"] = "hit" if ok else "miss"
        if ok:
            hits.append(name)

        checked += 1

    return hits, detail


def derive_default_output_path(in_path: str) -> str:
    base, ext = os.path.splitext(in_path)
    if not ext:
        ext = ".json"
    return f"{base}_wikidata_check{ext}"


def run(in_path: str, out_path: str) -> None:
    t0 = time.time()
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; WikidataAnswerChecker/1.2; +https://www.wikidata.org/)"
    })

    qid_cache: Dict[str, Optional[str]] = {}
    html_cache: Dict[str, str] = {}

    total_q = len(data)
    hit_origin_cnt = 0
    hit_entity_cnt = 0

    log(f"Start. input={in_path}")
    log(f"Total questions={total_q}, STRICT_BY_ANSWER_QID={STRICT_BY_ANSWER_QID}, HIT_AS_LIST={HIT_AS_LIST}")

    outputs = []
    for i, item in enumerate(data, start=1):
        qid = item.get("Id")
        answer_labels, answer_qids = extract_answer_labels_and_qids(item)

        origin_list = item.get("origin_entity", []) or []
        entity_list = item.get("entity", []) or []

        # 问题级进度日志
        if i == 1 or i % LOG_EVERY_Q == 0 or i == total_q:
            elapsed = time.time() - t0
            log(f"Progress {i}/{total_q} (elapsed {elapsed:.1f}s). Current Id={qid} "
                f"| hits(origin/entity)={hit_origin_cnt}/{hit_entity_cnt} "
                f"| cache(qid/html)={len(qid_cache)}/{len(html_cache)}")

        origin_hits, origin_detail = check_entity_list_for_answer(
            session, qid, "origin_entity", origin_list, answer_labels, answer_qids, qid_cache, html_cache
        )
        entity_hits, entity_detail = check_entity_list_for_answer(
            session, qid, "entity", entity_list, answer_labels, answer_qids, qid_cache, html_cache
        )

        if origin_hits:
            hit_origin_cnt += 1
        if entity_hits:
            hit_entity_cnt += 1

        if HIT_AS_LIST:
            origin_ret = origin_hits if origin_hits else "failed"
            entity_ret = entity_hits if entity_hits else "failed"
        else:
            origin_ret = origin_hits[0] if origin_hits else "failed"
            entity_ret = entity_hits[0] if entity_hits else "failed"

        outputs.append({
            "Id": qid,
            "origin_entity_retrival": origin_ret,
            "entity_retrival": entity_ret,

            # 为了你审计“是否真的跳转到 Qxxx 并检索”，保留明细（你不想要可删）
            "answer_labels": answer_labels,
            "answer_qids": answer_qids,
            "origin_entity_detail": origin_detail,
            "entity_detail": entity_detail,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log(f"Done. output={out_path}")
    log(f"Summary: total={total_q}, hit_origin={hit_origin_cnt}, hit_entity={hit_entity_cnt}, "
        f"cache(qid/html)={len(qid_cache)}/{len(html_cache)}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    if not output_path:
        output_path = derive_default_output_path(input_path)
    run(input_path, output_path)