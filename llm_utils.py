
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
from .utils import prompt_hash, records_sig, parse_pred, validate_and_normalize_pred, get_pred_label

ASYNC_CONCURRENCY = 8
# ========= 提示生成与精修（加入硬性字段要求） =========
def gen_initial_prompts(client: OpenAI, model: str, batches: List[List[Tuple[List[Dict], int]]], initial_prompt_seed: str,
                        label_field: str, num_candidates_per_batch=2):
    prompts = []
    sys = f"You will propose concise system prompts that improve 0/1 prediction accuracy for {label_field} given k-1 check-ups."
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
            f"Given these examples, propose a system prompt that maximizes correctness of {label_field} (0/1). "
            "Constraints: JSON-only output; evidence-based; no assumptions; short but precise rules.\n"
            "Hard requirement: The optimized system prompt must force the model to output STRICT JSON with ALL of these keys present exactly as in the seed: "
            f"Trend, Reasons, Most_important_indicator, Confidence, Predicted_subtype, {label_field}.\n"
            f"Seed prompt:\n{initial_prompt_seed}\n\n"
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
    prompts.append(initial_prompt_seed)
    return prompts

# ========= 异步预测（带指数退避 + 字段校验 + 重试 + 缓存） =========
async def _async_call_predict_one(async_client: AsyncOpenAI, model: str, p_sys: str, prefix_records: List[Dict],
                                  PRED_CACHE: Dict[Tuple[str, str], Dict], REQUIRED_KEYS: List[str], RESPONSE_SCHEMA: Dict,
                                  label_field: str,
                                  use_cache: bool = True, use_schema=True):

    key = (prompt_hash(p_sys), records_sig(prefix_records))
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
                response_format={"type": "json_schema",
                                 "json_schema": RESPONSE_SCHEMA} if use_schema
                                else {"type":"json_object"},
                timeout=60,
            )
            raw = resp.choices[0].message.content or ""
            data = parse_pred(raw) or {}
            data, missing = validate_and_normalize_pred(data,
                                                        REQUIRED_KEYS=REQUIRED_KEYS,
                                                        label_field=label_field)

            if not missing:
                if use_cache:
                    PRED_CACHE[key] = data
                return data

            # 缺字段 -> 构造一次纠错追问并继续下一轮
            err = (
                "Your previous output missed required JSON keys or had invalid values. "
                f"Missing/invalid: {missing}. "
                "Return STRICT JSON with ALL keys present exactly as: "
                f"Trend, Reasons, Most_important_indicator, Confidence, Predicted_subtype, {label_field}. "
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



# ========= 主流程：Prompt Auto-Tuning（评估与预测并发化，采样不变） =========
async def _async_predict_labels(p_sys: str, batch_samples: List[Tuple[List[Dict], int, Dict]], model: str,
                                PRED_CACHE: Dict[Tuple[str, str], Dict], REQUIRED_KEYS: List[str],
                                RESPONSE_SCHEMA: Dict,
                                label_field: str,
                                ):
    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

    async with AsyncOpenAI() as async_client:
        async def _task(prefix, gold, meta):
            async with sem:
                d = await _async_call_predict_one(async_client, model, p_sys, prefix,
                                                  PRED_CACHE, REQUIRED_KEYS, RESPONSE_SCHEMA,
                                                  label_field)
                return (prefix, gold, meta, get_pred_label(d, label_field))

        results = await asyncio.gather(*[_task(x, y, m) for (x, y, m) in batch_samples])

    return results

# ========= 数据集全量异步评估（保持采样数量不变 = 用完整 dataset） =========
async def _async_evaluate_prompt_full(p_sys: str, dataset: List[Tuple[List[Dict], int]], model: str,
                                      PRED_CACHE: Dict[Tuple[str, str], Dict], REQUIRED_KEYS: List[str],
                                      RESPONSE_SCHEMA: Dict,
                                      label_field: str,
                                      ) -> float:
    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

    async with AsyncOpenAI() as async_client:
        async def _task(prefix, gold):
            async with sem:
                d = await _async_call_predict_one(async_client, model, p_sys, prefix,
                                                  PRED_CACHE, REQUIRED_KEYS, RESPONSE_SCHEMA,
                                                  label_field)
                pred = get_pred_label(d, label_field)
                if pred is None:
                    return 0, 0
                return int(pred == gold), 1

        results = await asyncio.gather(*[_task(x, y) for (x, y) in dataset])

    correct = sum(c for c, t in results)
    total = sum(t for c, t in results)
    return (correct / total) if total > 0 else 0.0