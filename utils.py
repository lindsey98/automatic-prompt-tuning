import json
from typing import Dict, List, Tuple
import hashlib

# ========= 工具函数 =========
def drop_fields(record, drop_keys):
    return {k: v for k, v in record.items() if k not in drop_keys}

def sort_records(records):
    """按 Check-up ID 排序，并保证是 int。"""
    def _parse_id(x):
        v = x.get("Check-up ID", None)
        try:
            return int(v)
        except Exception:
            return v
    return sorted(records, key=_parse_id)

def parse_pred(raw: str) -> Dict:
    # 首选严格 JSON 解析
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        # 容错：提取第一个大括号块
        s = raw.find("{"); e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e+1])
            except:
                return {}
        return {}

def get_pred_label(data: Dict, label_field: str):
    v = data.get(label_field, None)
    try:
        return int(v) if v is not None else None
    except:
        return None

def prompt_hash(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:12]

def records_sig(records: List[Dict]) -> str:
    s = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def validate_and_normalize_pred(data: Dict, REQUIRED_KEYS: List[str], label_field: str) -> Tuple[Dict, List[str]]:
    """
    返回 (规范化后的data, 缺失字段列表)。
    温和规范化：Reasons 保证为 list[str]；Confidence 归一到 High/Medium/Low；
    Fatty_Liver 保证是 0/1 的 int；其他字段只检查存在性。
    """
    if not isinstance(data, dict):
        return {}, REQUIRED_KEYS

    missing = [k for k in REQUIRED_KEYS if k not in data]
    out = dict(data)

    # Reasons -> list[str]
    if "Reasons" in out:
        if isinstance(out["Reasons"], str):
            out["Reasons"] = [out["Reasons"]]
        elif not isinstance(out["Reasons"], list):
            out["Reasons"] = []
        else:
            out["Reasons"] = [str(x) for x in out["Reasons"]]

    # Confidence -> High/Medium/Low
    if "Confidence" in out:
        cv = str(out["Confidence"]).strip().lower()
        if "high" in cv:
            out["Confidence"] = "High"
        elif "low" in cv:
            out["Confidence"] = "Low"
        else:
            out["Confidence"] = "Medium"

    # Fatty_Liver -> int 0/1
    if label_field in out:
        try:
            out[label_field] = int(out[label_field])
            out[label_field] = 1 if out[label_field] == 1 else 0
        except Exception:
            if label_field not in missing:
                missing.append(label_field)

    return out, missing



# ========= 构造样本：每条为(前缀记录列表, 第k次真实标签, meta) =========
def make_samples_from_grouped(grouped_dict_list, label_field: str, min_k=3, drop_keys=None):
    samples = []  # (prefix_records, gold_label, meta)
    drop_keys = drop_keys or []
    for patient in grouped_dict_list:
        if not patient: continue
        ps = sort_records(patient)
        for k in range(min_k, len(ps)+1):
            prefix = [drop_fields(rec, drop_keys) for rec in ps[:k-1]]
            gold = ps[k-1].get(label_field, None)
            if gold is None:
                continue
            try:
                gold = int(gold)
            except:
                continue
            samples.append((prefix, gold, {"ID": ps[0].get("ID", None), "k": k}))
    return samples


