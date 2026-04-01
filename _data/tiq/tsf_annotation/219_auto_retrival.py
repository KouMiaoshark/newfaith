import json
import time
import random
import os
import requests
from typing import Any, Dict, List, Optional, Tuple

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_PAGE = "https://www.wikidata.org/wiki/{}"

# ========= 只需改这里 =========
input_path = r"历史提示词处理/simplify_rela_entity_answer_dev_solution9_implicit_6001-6301_dev.json"
# output_path = None
output_path = r"output_6001-6301_check.json"  # 也可以手动指定
# =================================

SEARCH_LANGUAGE = "en"
SEARCH_LIMIT = 5
REQUEST_TIMEOUT = 20
SLEEP_RANGE = (0.15, 0.35)
MAX_RETRY = 5
HIT_AS_LIST = True  # True: 输出命中实体名列表；False: 只输出第一个命中实体名


def _sleep():
    time.sleep(random.uniform(*SLEEP_RANGE))


def _request_with_retry(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                wait = min(2 ** attempt, 20) + random.uniform(0, 0.5)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == MAX_RETRY:
                raise
            wait = min(2 ** attempt, 20) + random.uniform(0, 0.5)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def wikidata_search_qid(
    session: requests.Session,
    query: str,
    language: str = SEARCH_LANGUAGE,
    limit: int = SEARCH_LIMIT,
) -> Optional[str]:
    """用 Wikidata 搜索实体名，返回最相关的 QID（Qxxx）。"""
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
    # 默认取第一个最相关结果
    return results[0].get("id")


def fetch_entity_page_html(session: requests.Session, qid: str) -> str:
    url = WIKIDATA_ENTITY_PAGE.format(qid)
    resp = _request_with_retry(session, "GET", url)
    return resp.text


def page_contains_any_answer_label(html: str, answer_labels: List[str]) -> Tuple[bool, List[str]]:
    h = html.lower()
    hit_labels = []
    for lab in answer_labels:
        if not lab:
            continue
        if lab.lower() in h:
            hit_labels.append(lab)
    return (len(hit_labels) > 0), hit_labels


def check_entity_list_for_answer(
    session: requests.Session,
    entity_names: List[str],
    answer_labels: List[str],
    qid_cache: Dict[str, Optional[str]],
    html_cache: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    hits: List[str] = []
    detail: Dict[str, Any] = {}

    for name in entity_names:
        name = str(name).strip()
        if not name:
            continue

        if name in qid_cache:
            qid = qid_cache[name]
        else:
            _sleep()
            qid = wikidata_search_qid(session, name)
            qid_cache[name] = qid

        detail[name] = {"qid": qid, "hit_labels": [], "status": "searched"}

        if not qid:
            detail[name]["status"] = "no_qid"
            continue

        if qid in html_cache:
            html = html_cache[qid]
        else:
            _sleep()
            html = fetch_entity_page_html(session, qid)
            html_cache[qid] = html

        ok, hit_labels = page_contains_any_answer_label(html, answer_labels)
        detail[name]["hit_labels"] = hit_labels
        detail[name]["status"] = "hit" if ok else "miss"

        if ok:
            hits.append(name)

    return hits, detail


def extract_answer_labels(item: Dict[str, Any]) -> List[str]:
    labels = []
    for ans in item.get("Answer", []) or []:
        lab = ans.get("WikidataLabel")
        if lab is not None:
            labels.append(str(lab).strip())
    seen = set()
    out = []
    for x in labels:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def derive_default_output_path(in_path: str) -> str:
    base, ext = os.path.splitext(in_path)
    if not ext:
        ext = ".json"
    return f"{base}_wikidata_check{ext}"


def run(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; WikidataAnswerChecker/1.0; +https://www.wikidata.org/)"
    })

    qid_cache: Dict[str, Optional[str]] = {}
    html_cache: Dict[str, str] = {}

    outputs = []
    for item in data:
        qid = item.get("Id")
        answer_labels = extract_answer_labels(item)

        origin_list = item.get("origin_entity", []) or []
        entity_list = item.get("entity", []) or []

        origin_hits, _origin_detail = check_entity_list_for_answer(
            session, origin_list, answer_labels, qid_cache, html_cache
        )
        entity_hits, _entity_detail = check_entity_list_for_answer(
            session, entity_list, answer_labels, qid_cache, html_cache
        )

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
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if not output_path:
        output_path = derive_default_output_path(input_path)

    run(input_path, output_path)
    print(f"Done. Output saved to: {output_path}")