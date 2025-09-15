# -*- coding: utf-8 -*-

import pandas as pd
from typing import List, Dict, Tuple
import json
import random
import asyncio
import hashlib

try:
    from tqdm import tqdm
except Exception:
    # 兜底：tqdm 不可用时给个兼容包装
    def tqdm(x, **kwargs):
        return x

from openai import OpenAI, AsyncOpenAI

# ========== 并发与缓存设置 ==========
ASYNC_CONCURRENCY = 8   # 按需 4~16；遇到 429 再下调
PRED_CACHE: Dict[Tuple[str, str], Dict] = {}  # (prompt_hash, records_sig) -> parsed_pred

INIT_PROMPT_SEED_FBG = """You are an endocrinologist with over 10 years of experience.  
Analyze k-1 consecutive check-up records, predict k-th FGB status: normal FBG (status=0) or elevated FBG (status=1).  

Rules:  
- Use evidence from data only, no assumptions.  
- Explain why/why not status changed.
- Predict likely disease subtype.  
- OUTPUT MUST BE STRICT JSON with ALL keys present exactly as specified below.  

Output: JSON only
{
  "Trend": "<gradual increase / sudden increase / fluctuating / no clear trend>",
  "Reasons": [
    "Reason 1 with values",
    "Reason 2",
    "Reason 3"
  ],
  "Most_important_indicator": "<FBG / HbA1c / Weight / BMI / TG / HDL / LDL / Total Cholesterol / Blood Pressure / Age>",
  "Confidence": "<High / Medium / Low>",
  "Predicted_subtype": "<Obesity-related IR / Lipid-related IR / Age-related / Mixed metabolic / Glucose-dominant / Other>",
  "Elevated_FBG": <0 or 1>
}
"""


# ========= 硬性字段约束 =========
REQUIRED_KEYS = [
    "Trend",
    "Reasons",
    "Most_important_indicator",
    "Confidence",
    "Predicted_subtype",
    "Elevated_FBG",
]

RESPONSE_SCHEMA = {
    "name": "ElevatedFBGReport",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Trend": {"type": "string"},
            "Reasons": {"type": "array", "items": {"type": "string"}},
            "Most_important_indicator": {"type": "string"},
            "Confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "Predicted_subtype": {"type": "string"},
            "Elevated_FBG": {"type": "integer", "enum": [0, 1]}
        },
        "required": ["Trend","Reasons","Most_important_indicator","Confidence","Predicted_subtype","Elevated_FBG"]
    },
    "strict": True
}

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

def get_pred_label(data: Dict):
    v = data.get("Elevated_FBG", None)
    try:
        return int(v) if v is not None else None
    except:
        return None

def _prompt_hash(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:12]

def _records_sig(records: List[Dict]) -> str:
    s = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def validate_and_normalize_pred(data: Dict) -> Tuple[Dict, List[str]]:
    """
    返回 (规范化后的data, 缺失字段列表)。
    温和规范化：Reasons 保证为 list[str]；Confidence 归一到 High/Medium/Low；
    Elevated_FBG 保证是 0/1 的 int；其他字段只检查存在性。
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

    # Elevated_FBG -> int 0/1
    if "Elevated_FBG" in out:
        try:
            out["Elevated_FBG"] = int(out["Elevated_FBG"])
            out["Elevated_FBG"] = 1 if out["Elevated_FBG"] == 1 else 0
        except Exception:
            if "Elevated_FBG" not in missing:
                missing.append("Elevated_FBG")

    return out, missing

# ========= 同步预测（带字段校验与重试） =========
def call_predict_one(client: OpenAI, model: str, p_sys: str, prefix_records: List[Dict], max_retry=2, use_schema=True):
    key = (_prompt_hash(p_sys), _records_sig(prefix_records))
    if key in PRED_CACHE:
        return PRED_CACHE[key]

    messages = [
        {"role": "developer", "content": p_sys},
        {"role": "user", "content": json.dumps(prefix_records, ensure_ascii=False)}
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.01,
        "response_format": {"type": "json_schema", "json_schema": RESPONSE_SCHEMA} if use_schema
                           else {"type": "json_object"},
    }

    for attempt in range(1, max_retry + 2):
        resp = client.chat.completions.create(**kwargs)
        raw  = resp.choices[0].message.content or ""
        data = parse_pred(raw) or {}
        data, missing = validate_and_normalize_pred(data)

        if not missing:
            PRED_CACHE[key] = data
            return data

        if attempt <= max_retry:
            err = (
                "Your previous output missed required JSON keys or had invalid values. "
                f"Missing/invalid: {missing}. "
                "Return STRICT JSON with ALL keys present exactly as: "
                "Trend, Reasons, Most_important_indicator, Confidence, Predicted_subtype, Elevated_FBG. "
                "Do not include any extra text."
            )
            kwargs["messages"] = [
                {"role": "developer", "content": p_sys},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": err},
            ]

    PRED_CACHE[key] = {}
    return {}

# ========= 异步预测（带指数退避 + 字段校验 + 重试 + 缓存） =========
async def _async_call_predict_one(async_client: AsyncOpenAI, model: str, p_sys: str, prefix_records: List[Dict], use_cache: bool = True, use_schema=True):
    key = (_prompt_hash(p_sys), _records_sig(prefix_records))
    if use_cache and key in PRED_CACHE:
        return PRED_CACHE[key]

    base_messages = [
        {"role": "developer", "content": p_sys.strip()},
        {"role": "user", "content": json.dumps(prefix_records, ensure_ascii=False)}
    ]

    messages = list(base_messages)
    for attempt in range(3):  # 轻量重试 3 次
        try:
            resp = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.01,
                response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA} if use_schema
                                else {"type":"json_object"},
                timeout=60,
            )
            raw = resp.choices[0].message.content or ""
            data = parse_pred(raw) or {}
            data, missing = validate_and_normalize_pred(data)

            if not missing:
                if use_cache:
                    PRED_CACHE[key] = data
                return data

            # 缺字段 -> 构造一次纠错追问并继续下一轮
            err = (
                "Your previous output missed required JSON keys or had invalid values. "
                f"Missing/invalid: {missing}. "
                "Return STRICT JSON with ALL keys present exactly as: "
                "Trend, Reasons, Most_important_indicator, Confidence, Predicted_subtype, Elevated_FBG. "
                "Do not include any extra text."
            )
            messages = [
                {"role": "developer", "content": p_sys.strip()},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": err},
            ]
        except Exception:
            backoff = min(2 ** attempt, 8)
            await asyncio.sleep(backoff)

    if use_cache:
        PRED_CACHE[key] = {}
    return {}

# ========= 数据集全量异步评估（保持采样数量不变 = 用完整 dataset） =========
async def _async_evaluate_prompt_full(p_sys: str, dataset: List[Tuple[List[Dict], int]], model: str) -> float:
    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

    async with AsyncOpenAI() as async_client:
        async def _task(prefix, gold):
            async with sem:
                d = await _async_call_predict_one(async_client, model, p_sys, prefix)
                pred = get_pred_label(d)
                if pred is None:
                    return 0, 0
                return int(pred == gold), 1

        results = await asyncio.gather(*[_task(x, y) for (x, y) in dataset])

    correct = sum(c for c, t in results)
    total = sum(t for c, t in results)
    return (correct / total) if total > 0 else 0.0


def evaluate_prompt(model: str, p_sys: str, dataset: List[Tuple[List[Dict], int]]):
    return asyncio.run(_async_evaluate_prompt_full(p_sys, dataset, model))

# ========= 构造样本：每条为(前缀记录列表, 第k次真实标签, meta) =========
def make_samples_from_grouped(grouped_dict_list, min_k=3, drop_keys=None):
    samples = []  # (prefix_records, gold_label, meta)
    drop_keys = drop_keys or []
    for patient in grouped_dict_list:
        if not patient: continue
        ps = sort_records(patient)
        for k in range(min_k, len(ps)+1):
            prefix = [drop_fields(rec, drop_keys) for rec in ps[:k-1]]
            gold = ps[k-1].get("Elevated_FBG", None)
            if gold is None:
                continue
            try:
                gold = int(gold)
            except:
                continue
            samples.append((prefix, gold, {"ID": ps[0].get("ID", None), "k": k}))
    return samples

# ========= 提示生成与精修（加入硬性字段要求） =========
def gen_initial_prompts(client: OpenAI, model: str, batches: List[List[Tuple[List[Dict], int]]], num_candidates_per_batch=2):
    prompts = []
    sys = "You will propose concise system prompts that improve 0/1 prediction accuracy for Elevated_FBG given k-1 check-ups."
    for bk in tqdm(batches):
        examples = []
        for (prefix, gold, meta) in bk[:4]:
            ex = {
                "k": meta.get("k"),
                "records": prefix,
                "gold": gold
            }
            examples.append(ex)
        user = (
            "Given these examples, propose a system prompt that maximizes correctness of Elevated_FBG (0/1). "
            "Constraints: JSON-only output; evidence-based; no assumptions; short but precise rules.\n"
            "Hard requirement: The optimized system prompt must force the model to output STRICT JSON with ALL of these keys present exactly as in the seed: "
            "Trend, Reasons, Most_important_indicator, Confidence, Predicted_subtype, Elevated_FBG.\n"
            f"Seed prompt:\n{INIT_PROMPT_SEED_FBG}\n\n"
            f"Examples:\n{json.dumps(examples, ensure_ascii=False, indent=2)}\n\n"
            "Return ONLY the system prompt text."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user}],
            n=num_candidates_per_batch,
            temperature=1.0,
        )
        for ch in resp.choices:
            ptxt = (ch.message.content or "").strip()
            if ptxt:
                prompts.append(ptxt)
    prompts.append(INIT_PROMPT_SEED_FBG)
    return prompts

def refine_prompts_from_feedback_FBG(
    client: OpenAI,
    model: str,
    p_star: str,
    batch: List[Tuple[List[Dict], int]],
    num_candidates=3
):
    examples = []
    for (prefix, gold, meta) in batch[:6]:
        ex = {"k": meta["k"], "records": prefix, "gold": gold}
        examples.append(ex)

    fb_sys = "You are optimizing a system prompt for accurate 0/1 Elevated_FBG prediction."
    fb_user = (
        "Current prompt:\n"
        f"{p_star}\n\n"
        "Analyze common misclassifications and propose concrete, succinct rules or ordering of criteria to fix them. "
        "Focus on: FBG↑ trends across visits, HbA1c↑, Glycated Albumin↑, Weight↑ or BMI↑, TG↑, HDL↓, Age↑. "
        "Handle noisy fluctuations by looking at consistent upward trends instead of single-visit spikes. "
        "Avoid over-relying on just one abnormal value; consider multiple markers together.\n\n"
        f"Examples (JSON):\n"
        f"{json.dumps(examples, ensure_ascii=False, indent=2)}\n\n"
        "Return a bullet list of suggested changes."
    )
    fb_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":fb_sys},
            {"role":"user","content":fb_user}],
        temperature=1.0,
    )
    feedback = fb_resp.choices[0].message.content or ""

    rw_sys = "Rewrite the system prompt for Elevated_FBG prediction to improve accuracy. Keep it compact and enforce STRICT JSON output."
    rw_user = (
        f"Original prompt:\n{p_star}\n\n"
        f"Feedback (apply all applicable):\n{feedback}\n\n"
        "Hard requirement: The rewritten system prompt must force the model to output STRICT JSON with ALL of these keys present exactly: "
        "Trend, Reasons, Most_important_indicator, Confidence, Predicted_subtype, Elevated_FBG.\n"
        "Return ONLY the new system prompt."
    )
    rw_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":rw_sys},
            {"role":"user","content":rw_user}
        ],
        n=num_candidates,
        temperature=1.0,
    )
    candidates = []
    for ch in rw_resp.choices:
        txt = (ch.message.content or "").strip()
        if txt:
            candidates.append(txt)
    return candidates


# ========= 主流程：Prompt Auto-Tuning（评估与预测并发化，采样不变） =========
async def _async_predict_labels(p_sys: str, batch_samples: List[Tuple[List[Dict], int, Dict]], model: str):
    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

    async with AsyncOpenAI() as async_client:
        async def _task(prefix, gold, meta):
            async with sem:
                d = await _async_call_predict_one(async_client, model, p_sys, prefix)
                return (prefix, gold, meta, get_pred_label(d))

        results = await asyncio.gather(*[_task(x, y, m) for (x, y, m) in batch_samples])

    return results

def prompt_auto_tune(
    grouped_dict_list: List[List[Dict]],
    model_eval="gpt-4o-mini",
    seed=0,
    train_ratio=0.8,
    batch_size=8,
    epochs=3,
    num_init_cands_per_batch=2,
    num_refine_cands=3,
):

    random.seed(seed)
    client = OpenAI()

    # --- 构造样本 ---
    samples_all = make_samples_from_grouped(grouped_dict_list,
                                            min_k=3,
                                            drop_keys=["Reasons","Reasons_str","Predicted_subtype","Elevated_FBG"])
    random.shuffle(samples_all)

    # train/valid split
    split = int(len(samples_all)*train_ratio)
    train = samples_all[:split]; valid = samples_all[split:]
    train_ds = [(x, y) for (x,y,_) in train]
    valid_ds = [(x, y) for (x,y,_) in valid]

    # --- 初始分批 ---
    batches = []
    for i in range(0, len(train), batch_size):
        batches.append(train[i:i+batch_size])

    # --- 初始候选提示 ---
    init_prompts = gen_initial_prompts(client, model_eval, batches, num_candidates_per_batch=num_init_cands_per_batch)

    # --- 选择初始 p*（全量并发评估；采样不变）---
    scores = []
    for p in init_prompts:
        acc = evaluate_prompt(model_eval, p, train_ds)  # 忽略 sample_limit -> 全量
        scores.append((p, acc))
    p_star, best_score = max(scores, key=lambda x: x[1])
    print(f"[Init] best prompt={p_star}")
    print(f"[Init] best train acc={best_score:.4f}")

    # ========= 迭代优化 =========
    for e in range(1, epochs+1):
        # 1) 用 p* 在训练集预测（并发），划分 S+ / S-
        pred_triplets = asyncio.run(_async_predict_labels(p_star, train, model_eval))
        S_pos, S_neg = [], []
        for (x, y, meta, pred) in pred_triplets:
            if pred is None or pred != y:
                S_neg.append((x, y, meta))
            else:
                S_pos.append((x, y, meta))

        # 2) 重新成批：确保每批含 S+ 与 S-
        mixed = []
        pos_i = 0; neg_i = 0
        while pos_i < len(S_pos) or neg_i < len(S_neg):
            batch = []
            for _ in range(batch_size):
                if neg_i < len(S_neg):
                    batch.append(S_neg[neg_i]); neg_i += 1
                elif pos_i < len(S_pos):
                    batch.append(S_pos[pos_i]); pos_i += 1
            if batch:
                mixed.append(batch)

        # 3) 对每个混合批：生成反馈 -> 重写成候选提示
        cand_pool = [p_star]  # 保留当前最优
        for bk in mixed:
            cands = refine_prompts_from_feedback_FBG(client, model_eval, p_star, bk, num_candidates=num_refine_cands)
            cand_pool.extend(cands)

        # 4) 评估候选，选新的 p*（全量并发评估）
        scored = []
        for p in cand_pool:
            acc = evaluate_prompt(model_eval, p, train_ds)
            scored.append((p, acc))
        p_star, best_score = max(scored, key=lambda x: x[1])
        print(f"[Epoch {e}] best prompt={p_star}")
        print(f"[Epoch {e}] best train acc={best_score:.4f}")

    # --- 最终在验证集评估（若有） ---
    val_acc = evaluate_prompt(model_eval, p_star, valid_ds) if valid_ds else None
    if val_acc is not None:
        print(f"[Final] valid acc={val_acc:.4f}")
    else:
        print("[Final] no valid split")

    return p_star, best_score, val_acc


# ========= 额外新增：批量预测 + 合并保存 =========
def build_prediction_df(grouped_dict_list: List[List[Dict]],
                        p_sys: str,
                        model: str,
                        min_k: int = 3,
                        drop_keys=None,
                        reasons_joiner: str = " | ") -> pd.DataFrame:
    """
    对所有患者的第 k 次就诊（k>=min_k）生成预测，并返回包含键 ['ID','Check-up ID'] 的宽表用于合并。
    """
    drop_keys = drop_keys or []
    jobs = []
    for patient in grouped_dict_list:
        if not patient:
            continue
        ps = sort_records(patient)
        pid = ps[0].get("ID", None)
        for k in range(min_k, len(ps) + 1):
            prefix = [drop_fields(rec, drop_keys) for rec in ps[:k-1]]
            target = ps[k-1]
            jobs.append({"ID": pid, "Check-up ID": target.get("Check-up ID"), "prefix": prefix})

    async def _runner():
        sem = asyncio.Semaphore(ASYNC_CONCURRENCY)
        async with AsyncOpenAI() as async_client:
            async def _one(job):
                async with sem:
                    d = await _async_call_predict_one(async_client, model, p_sys, job["prefix"])
                    d = d or {}
                    reasons = d.get("Reasons") or []
                    if isinstance(reasons, list):
                        reasons_str = reasons_joiner.join([str(x) for x in reasons])
                    else:
                        reasons_str = str(reasons)
                    return {
                        "ID": job["ID"],
                        "Check-up ID": job["Check-up ID"],
                        "Trend": d.get("Trend"),
                        "Reasons": reasons_str,
                        "Most_important_indicator": d.get("Most_important_indicator"),
                        "Confidence": d.get("Confidence"),
                        "Predicted_subtype": d.get("Predicted_subtype"),
                        "Elevated_FBG": d.get("Elevated_FBG"),
                    }
            tasks = [_one(j) for j in jobs]
            return await asyncio.gather(*tasks)

    results = asyncio.run(_runner())
    df_pred = pd.DataFrame(results)
    return df_pred

def save_merged_predictions_overwrite(df_original: pd.DataFrame,
                                      df_pred: pd.DataFrame,
                                      out_csv_path: str,
                                      keys=None,
                                      target_cols=None) -> pd.DataFrame:
    """
    与原始 df 合并，但不丢任何列；用 df_pred 的同名列覆盖原 df 中的列。
    仅在 df_pred 对应值非空时覆盖；df_pred 若缺某行/列则保留原值。
    """
    if keys is None:
        keys = ["ID", "Check-up ID"]
    if target_cols is None:
        target_cols = ["Trend","Reasons","Most_important_indicator","Confidence","Predicted_subtype","Elevated_FBG"]

    # 只取用于覆盖的列 + 键
    need_cols = [c for c in target_cols if c in df_pred.columns]
    use_cols = list(dict.fromkeys(keys + need_cols))  # 去重保序
    pred_use = df_pred[use_cols].copy()

    # set_index + update：不会新增列名，且仅覆盖非空值
    out = df_original.copy()
    out_idx = out.set_index(keys)
    pred_idx = pred_use.set_index(keys)

    # 只保留将要覆盖的目标列，避免 update 引入多余列
    pred_idx = pred_idx[need_cols]

    # 执行覆盖（pred 非空才覆盖）
    out_idx.update(pred_idx)

    out = out_idx.reset_index()
    out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return out

# ========= 入口 =========
if __name__ == '__main__':
    # 你的 CSV 文件路径
    df = pd.read_csv("Diabetes_60samples.csv")
    out_csv = "Diabetes_60samples_with_model_outputs_merged.csv"
    model_name = "gpt-4o-mini"

    keep_cols = [
        "ID",
        "Check-up ID",
        "Blood_Glucose",
        "HbA1c",
        "Glycated_Albumin",
        "Weight",
        "Height",  # 用来计算 BMI
        "Blood_Pressure",
        "Triglycerides",
        "Total_Cholesterol",
        "LDL",
        "HDL",
        "Apolipoprotein_A1",
        "Apolipoprotein_B",
        "Lipoprotein_a",
        "ALT",
        "AST",
        "GGT",
        "ALP",
        "Albumin",
        "Total_Protein",
        "Urea",
        "Creatinine",
        "eGFR_EPI",
        "Cystatin_C",
        "Uric_Acid",
        "Elevated_FBG"
    ]

    df_subset = df[keep_cols]
    grouped_dict_list = [group.to_dict(orient="records") for _, group in df_subset.groupby("ID")]

    p_star, train_acc, valid_acc = prompt_auto_tune(
        grouped_dict_list=grouped_dict_list,
        model_eval=model_name,
        seed=42,
        train_ratio=0.8,
        batch_size=8,
        epochs=10,
        num_init_cands_per_batch=2,
        num_refine_cands=5,
    )
    print("\n=== Best Prompt (p*) ===\n", p_star)

    samples_all = make_samples_from_grouped(grouped_dict_list,
                                            min_k=3,
                                            drop_keys=["Reasons","Reasons_str","Predicted_subtype", "Elevated_FBG"])
    full_ds = [(x, y) for (x,y,_) in samples_all]
    acc = evaluate_prompt("gpt-4o-mini", p_star, full_ds)
    print(f"[Final] Full acc={acc:.4f}")

    # ===== 新增：批量预测并合并保存
    df_pred = build_prediction_df(
        grouped_dict_list=grouped_dict_list,
        p_sys=p_star,
        model=model_name,
        min_k=3,
        drop_keys=["Reasons","Reasons_str","Predicted_subtype","Elevated_FBG"],  # 防泄漏
        reasons_joiner=" | "  # 将列表拼成可读字符串
    )

    target_cols = ["Trend","Reasons","Most_important_indicator","Confidence","Predicted_subtype","Elevated_FBG"]
    df_base = df.drop(target_cols, axis=1)
    df_merged = df_base.merge(df_pred, on=["ID", "Check-up ID"], how="left")
    df_merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
