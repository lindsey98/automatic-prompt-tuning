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
from utils import *
from llm_utils import _async_evaluate_prompt_full, _async_call_predict_one, _async_predict_labels, gen_initial_prompts

# ========== 并发与缓存设置 ==========
ASYNC_CONCURRENCY=8
PRED_CACHE: Dict[Tuple[str, str], Dict] = {}  # (prompt_hash, records_sig) -> parsed_pred
INIT_PROMPT_SEED = open("./fatty_liver_init_prompt.txt").read()

# ========= 硬性字段约束 =========
REQUIRED_KEYS = [
    "Trend",
    "Reasons",
    "Most_important_indicator",
    "Confidence",
    "Predicted_subtype",
    "Fatty_Liver",
]

RESPONSE_SCHEMA = {
    "name": "FattyLiverReport",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Trend": {"type": "string"},
            "Reasons": {"type": "array", "items": {"type": "string"}},
            "Most_important_indicator": {"type": "string"},
            "Confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "Predicted_subtype": {"type": "string"},
            "Fatty_Liver": {"type": "integer", "enum": [0, 1]}
        },
        "required": ["Trend","Reasons","Most_important_indicator","Confidence","Predicted_subtype","Fatty_Liver"]
    },
    "strict": True
}


def evaluate_prompt(model: str, p_sys: str, dataset: List[Tuple[List[Dict], int]]):
    return asyncio.run(_async_evaluate_prompt_full(p_sys, dataset, model,
                                                   PRED_CACHE, REQUIRED_KEYS, RESPONSE_SCHEMA,
                                                   label_field="Fatty_Liver"))


def refine_prompts_from_feedback(client: OpenAI, model: str, p_star: str, batch: List[Tuple[List[Dict], int]], num_candidates=3):
    examples = []
    for (prefix, gold, meta) in batch[:6]:
        ex = {"k": meta["k"], "records": prefix, "gold": gold}
        examples.append(ex)

    fb_sys = "You are optimizing a system prompt for accurate 0/1 fatty liver prediction."
    fb_user = (
        "Current prompt:\n"
        f"{p_star}\n\n"
        "Analyze typical errors and propose concrete, succinct rules or ordering of criteria to fix them. "
        "Focus on: TG↑, LDL/TC↑, HDL↓, Glucose↑, Weight↑ trends; handle noisy fluctuations; "
        "avoid over-relying on single visit.\n\n"
        f"Examples (JSON):\n"
        f"{json.dumps(examples, ensure_ascii=False, indent=2)}\n\n"
        "Return a bullet list of changes."
    )
    fb_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":fb_sys},
            {"role":"user","content":fb_user}],
        temperature=1.0,
    )
    feedback = fb_resp.choices[0].message.content or ""

    rw_sys = "Rewrite the system prompt to improve accuracy. Keep it compact and enforce STRICT JSON output."
    rw_user = (
        f"Original prompt:\n{p_star}\n\n"
        f"Feedback (apply all applicable):\n{feedback}\n\n"
        "Hard requirement: The rewritten system prompt must force the model to output STRICT JSON with ALL of these keys present exactly: "
        "Trend, Reasons, Most_important_indicator, Confidence, Predicted_subtype, Fatty_Liver.\n"
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
                                            label_field="Fatty_Liver",
                                            drop_keys=["Reasons","Reasons_str","Predicted_subtype","Fatty_Liver"])
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
    init_prompts = gen_initial_prompts(client, model_eval, batches,
                                       initial_prompt_seed=INIT_PROMPT_SEED,
                                       label_field="Fatty_Liver", num_candidates_per_batch=num_init_cands_per_batch)

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
        pred_triplets = asyncio.run(_async_predict_labels(p_star, train, model_eval,
                                                          PRED_CACHE=PRED_CACHE, REQUIRED_KEYS=REQUIRED_KEYS, RESPONSE_SCHEMA=RESPONSE_SCHEMA,
                                                          label_field="Fatty_Liver")
                                    )
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
            cands = refine_prompts_from_feedback(client, model_eval, p_star, bk, num_candidates=num_refine_cands)
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
                    d = await _async_call_predict_one(async_client, model, p_sys, job["prefix"],
                                                      PRED_CACHE=PRED_CACHE, REQUIRED_KEYS=REQUIRED_KEYS,
                                                      RESPONSE_SCHEMA=RESPONSE_SCHEMA,
                                                      label_field="Fatty_Liver"
                                                      )
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
                        "Fatty_Liver": d.get("Fatty_Liver"),
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
        target_cols = ["Trend","Reasons","Most_important_indicator","Confidence","Predicted_subtype","Fatty_Liver"]

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
    df = pd.read_csv("Fat_40sample_with_visit4_predictions.csv")
    out_csv = "Fat_40sample_with_model_outputs_merged.csv"
    model_name = "gpt-4o-mini"

    keep_cols = [
        "ID",
        "Check-up ID",
        "Gender",
        "Age",
        "Height",
        "Weight",
        "Blood_Glucose",
        "Total_Protein",
        "Triglycerides",
        "Total_Cholesterol",
        "LDL",
        "HDL",
        "ALT",
        "AST",
        "GGT",
        "ALP",
        "HbA1c",
        "Reasons",
        "Reasons_str",
        "Predicted_subtype",
        "Fatty_Liver"
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
    # p_star = """You are a hepatologist and endocrinologist with over 10 years of experience. Analyze k-1 consecutive check-up records. If records are fewer than three visits, return 'Fatty_Liver' as 0 and note insufficient data.
    # Critical thresholds: Triglycerides (TG > 1.7 mmol/L), LDL (LDL > 3.0 mmol/L), HDL (<1.0 mmol/L for men, <1.3 mmol/L for women), and Blood Glucose (>5.5 mmol/L). Identify weight trends using a median approach across visits, noting any increase >2 kg as critical and a decrease ≤2 kg as a minor concern.
    # Prioritize indicators in decision-making: TG, HDL, weight change.
    # Any LDL >3.0 mmol/L indicates increased concern regardless of other findings.
    # Handle null values by reporting as 'NaN' and note their impact on analyses.
    # Define confidence levels: High for consistent markers, Medium for some concerns, Low for misalignment across visits.
    # Clarify subtype logic based on weight and lipid profiles: Obesity-related, Hyperlipidemia-related, ALT-dominant, Mixed metabolic.
    # OUTPUT MUST BE STRICT JSON with ALL keys present exactly as specified below.",
    # "output_format": {
    #     "Trend": "<gradual increase / sudden increase / fluctuating / no clear trend>",
    #     "Reasons": [
    #       "Reason 1 with values",
    #       "Reason 2",
    #       "Reason 3"
    #     ],
    #     "Most_important_indicator": "<BMI / ALT / TG / ultrasound finding / you name it>",
    #     "Confidence": "<High / Medium / Low>",
    #     "Predicted_subtype": "<Obesity-related / Hyperlipidemia-related / ALT-dominant / Mixed metabolic / you name it>",
    #     "Fatty_Liver": <0 or 1>
    # }
    # """
    samples_all = make_samples_from_grouped(grouped_dict_list,
                                            min_k=3,
                                            label_field="Fatty_Liver",
                                            drop_keys=["Reasons","Reasons_str","Predicted_subtype", "Fatty_Liver"])
    full_ds = [(x, y) for (x,y,_) in samples_all]
    acc = evaluate_prompt("gpt-4o-mini", p_star, full_ds)
    print(f"[Final] Full acc={acc:.4f}")

    # ===== 新增：批量预测并合并保存
    df_pred = build_prediction_df(
        grouped_dict_list=grouped_dict_list,
        p_sys=p_star,
        model=model_name,
        min_k=3,
        drop_keys=["Reasons","Reasons_str","Predicted_subtype","Fatty_Liver"],  # 防泄漏
        reasons_joiner=" | "  # 将列表拼成可读字符串
    )

    target_cols = ["Trend","Reasons","Most_important_indicator","Confidence","Predicted_subtype","Fatty_Liver"]
    df_base = df.drop(target_cols, axis=1)
    df_merged = df_base.merge(df_pred, on=["ID", "Check-up ID"], how="left")
    df_merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
