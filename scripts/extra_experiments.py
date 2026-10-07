#!/usr/bin/env python3
"""Supplementary experiments run AFTER the core NB1-NB5 verdict was recorded.

None of these touch the frozen baselines, the eval sets, `OPTIMIZED_PROMPT` or the
four graded rows of `results/runs.csv`. They write their own artefacts:

    python scripts/extra_experiments.py probe    # results/regression_probe.json
    python scripts/extra_experiments.py qual     # results/qualitative_compare.json
    python scripts/extra_experiments.py replay   # results/replay.json   + adapters/replay_*/
    python scripts/extra_experiments.py sweep    # results/rank_sweep.json + adapters/sweep_r*/

probe   NB5 scores the regression set but does not keep the outputs. This saves them
        per item for the base model and the `correct` adapter, so the FAILED verdict
        can be diagnosed from evidence instead of from the aggregate.
replay  Deck §6.3 fix for catastrophic forgetting: retrain the `correct` config with a
        small share of general Q&A mixed in. The replay answers are SELF-DISTILLED (the
        base model's own greedy answers), the questions are hand-written and disjoint
        from data/eval_regression.jsonl. Same step budget as `correct`.
sweep   Bonus B4: placement fixed at text-linear, r in {8, 16, 64}. r=16 is the
        `correct` adapter itself and is not retrained.

All extra training runs log to results/extra_runs.csv, never to runs.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, evaluate as ev, generate, modeling, report, train  # noqa: E402
from labkit.config import SPECS, get_tier, training_epochs  # noqa: E402

TIER = get_tier()
RESULTS = ROOT / "results"
SEED = 42

# Hand-written general questions. Checked by hand to share no question and no graded
# keyword with data/eval_regression.jsonl -- training on the eval set would make the
# regression score meaningless. `check_disjoint()` re-checks the exact-text part.
REPLAY_QUESTIONS = [
    "Một tuần có bao nhiêu ngày?",
    "Thủ đô của Nhật Bản là thành phố nào?",
    "15 cộng 27 bằng bao nhiêu?",
    "Viết một câu cảm ơn thầy cô nhân ngày Nhà giáo Việt Nam.",
    "Dịch sang tiếng Anh: 'Hôm nay trời đẹp'.",
    "Núi nào cao nhất Việt Nam?",
    "Hành tinh nào gần Mặt Trời nhất?",
    "Nêu một cách tiết kiệm điện trong gia đình.",
    "Một giờ có bao nhiêu phút?",
    "Nước đóng băng ở bao nhiêu độ C?",
    "Kể tên một loài động vật sống dưới biển.",
    "Giải thích ý nghĩa câu tục ngữ 'Uống nước nhớ nguồn'.",
    "Vịnh Hạ Long nằm ở tỉnh nào?",
    "Giải thích ngắn gọn trọng lực là gì.",
    "Kể tên ba màu cơ bản.",
    "9 nhân 8 bằng bao nhiêu?",
    "Viết một câu chào buổi sáng bằng tiếng Anh.",
    "Thủ đô của Pháp là gì?",
    "Kể tên hai nhạc cụ dân tộc Việt Nam.",
    "Hãy viết một câu giới thiệu bản thân ngắn.",
    "Trái Đất quay quanh thiên thể nào?",
    "Kể tên một môn thể thao phổ biến ở Việt Nam.",
    "Một năm nhuận có bao nhiêu ngày?",
    "Đỉnh núi cao nhất thế giới tên là gì?",
]


def load_jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


TARGET = load_jsonl(ROOT / "data" / "eval_target.jsonl")
REGRESSION = load_jsonl(ROOT / "data" / "eval_regression.jsonl")
TRAIN_ROWS = load_jsonl(ROOT / "data" / "split" / "train.jsonl")


def check_disjoint() -> None:
    eval_q = {ev.normalize(r["instruction"]) for r in REGRESSION}
    clash = [q for q in REPLAY_QUESTIONS if ev.normalize(q) in eval_q]
    assert not clash, f"replay questions overlap the regression eval set: {clash}"


def score(model, tok, *, with_regression: bool = True, label: str) -> tuple[dict, list, list]:
    """Exactly NB5's scoring recipe: NAIVE_PROMPT on target, no system prompt on regression."""
    preds, lat = generate.generate_batch(model, tok, [r["input"] for r in TARGET],
                                         system=generate.NAIVE_PROMPT, label=f"{label}/target")
    tgt = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, TARGET)) / len(TARGET)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
    rpreds, reg = [], None
    if with_regression:
        rpreds, _ = generate.generate_batch(model, tok, [r["instruction"] for r in REGRESSION],
                                            system=None, max_new_tokens=96,
                                            label=f"{label}/regression")
        reg = sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(rpreds, REGRESSION)) / len(REGRESSION)
    out = {"target": round(tgt, 4), "format": round(fmt, 4),
           "regression": None if reg is None else round(reg, 4),
           "latency_ms": round(lat, 1), "n": len(TARGET)}
    return out, preds, rpreds


def load_adapter(adapter_dir: pathlib.Path):
    from peft import PeftModel
    model, tok = generate.load_base(TIER)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tok


def looks_like_triage(pred: str) -> bool:
    obj = ev._parse_json_loose(pred)
    return isinstance(obj, dict) and any(k in obj for k in ev.TRIAGE_KEYS)


# --- probe ------------------------------------------------------------------

def cmd_probe(_args) -> None:
    out = {"model": TIER.model_id, "items": []}
    runs = {}
    model, tok = generate.load_base(TIER)
    runs["base"], _ = generate.generate_batch(model, tok, [r["instruction"] for r in REGRESSION],
                                              system=None, max_new_tokens=96, label="base/regression")
    del model
    generate.free_memory()
    model, tok = load_adapter(ROOT / "adapters" / "correct")
    runs["correct"], _ = generate.generate_batch(model, tok, [r["instruction"] for r in REGRESSION],
                                                 system=None, max_new_tokens=96,
                                                 label="correct/regression")
    del model
    generate.free_memory()

    for i, r in enumerate(REGRESSION):
        out["items"].append({
            "i": i, "instruction": r["instruction"], "keywords": r["keywords"],
            **{f"{k}_output": v[i] for k, v in runs.items()},
            **{f"{k}_recall": round(ev.keyword_recall(v[i], r["keywords"]), 4) for k, v in runs.items()},
            **{f"{k}_emits_triage_json": looks_like_triage(v[i]) for k, v in runs.items()},
        })
    for k in runs:
        out[f"{k}_regression"] = round(sum(x[f"{k}_recall"] for x in out["items"]) / len(REGRESSION), 4)
        out[f"{k}_triage_json_rate"] = round(
            sum(x[f"{k}_emits_triage_json"] for x in out["items"]) / len(REGRESSION), 4)
    report.write_json(out, "regression_probe.json", results_dir=RESULTS)
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, ensure_ascii=False, indent=2))
    for x in out["items"]:
        print(f"- {x['instruction'][:45]:45s} | base: {x['base_output'][:50]!r} | ft: {x['correct_output'][:60]!r}")


# --- shared training --------------------------------------------------------

def train_run(key: str, spec, records: list[dict], max_steps: int, extra: dict) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model, tok = generate.load_base(TIER, load_in_4bit=spec.load_in_4bit)
    ds = Dataset.from_list(data.to_training_dataset(tok, records, max_length=TIER.max_length,
                                                    mask_mode="assistant-only"))
    targets = modeling.resolve_target_modules(model, spec.target)
    trainable = modeling.count_lora_params(model, targets, spec.r)
    want = train.sft_config_kwargs(TIER, spec, str(ROOT / "adapters" / key), max_steps=max_steps)
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, want, label=f"SFTConfig[{key}]")
    lora_kwargs, _ = train.filter_kwargs(LoraConfig, train.lora_config_kwargs(spec, targets),
                                         label=f"LoraConfig[{key}]")
    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs), train_dataset=ds,
                         processing_class=tok, peft_config=LoraConfig(**lora_kwargs))
    train.align_trainable_precision(trainer.model)
    t0 = time.perf_counter()
    res = trainer.train()
    elapsed = time.perf_counter() - t0
    out = ROOT / "adapters" / key
    trainer.model.save_pretrained(out)

    row = train.summarize_run(spec, TIER, targets, trainable, elapsed, generate.peak_vram_gb())
    row.update(run=key, final_loss=round(res.training_loss, 4), max_steps=max_steps,
               n_train=len(ds), **extra)
    losses = [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h]
    report.append_row(row, "extra_runs.csv", results_dir=RESULTS)
    del trainer, model
    generate.free_memory()
    return {**row, "loss_curve": losses}


def correct_budget() -> int:
    rows = [r for r in report.read_rows("runs.csv", results_dir=RESULTS) if r.get("run") == "correct"]
    if rows and rows[-1].get("max_steps"):
        return int(rows[-1]["max_steps"])
    return train.planned_steps(len(TRAIN_ROWS), TIER, training_epochs())


# --- replay -----------------------------------------------------------------

def build_replay(n: int) -> list[dict]:
    """Self-distilled replay: the base model's own answers to general questions."""
    check_disjoint()
    cache = RESULTS / "replay_corpus.json"
    if cache.exists():
        corpus = json.loads(cache.read_text(encoding="utf-8"))
    else:
        model, tok = generate.load_base(TIER)
        answers, _ = generate.generate_batch(model, tok, REPLAY_QUESTIONS, system=None,
                                             max_new_tokens=160, label="replay/distill")
        del model
        generate.free_memory()
        corpus = [{"instruction": q, "input": "", "output": a}
                  for q, a in zip(REPLAY_QUESTIONS, answers)]
        report.write_json(corpus, "replay_corpus.json", results_dir=RESULTS)
    return corpus[:n]


def cmd_replay(args) -> None:
    import random
    budget = correct_budget()
    summary = {"model": TIER.model_id, "max_steps": budget, "runs": []}
    for pct in args.pct:
        n = max(1, math.ceil(len(TRAIN_ROWS) * pct / 100))
        if n > len(REPLAY_QUESTIONS):
            raise SystemExit(f"{pct}% needs {n} replay rows; only {len(REPLAY_QUESTIONS)} questions exist")
        replay = build_replay(n)
        mixed = TRAIN_ROWS + replay
        random.Random(SEED).shuffle(mixed)
        key = f"replay_{len(replay)}rows"
        print("=" * 70, f"\nRUN {key}: {len(TRAIN_ROWS)} task + {len(replay)} replay rows, {budget} steps")
        row = train_run(key, SPECS["correct"], mixed, budget,
                        {"label": f"correct + {len(replay)} self-distilled replay rows",
                         "replay_rows": len(replay), "replay_pct": pct})
        model, tok = load_adapter(ROOT / "adapters" / key)
        scores, _, rpreds = score(model, tok, label=key)
        del model
        generate.free_memory()
        summary["runs"].append({"run": key, "replay_rows": len(replay), **scores,
                                "final_loss": row["final_loss"], "loss_curve": row["loss_curve"],
                                "triage_json_on_regression": round(
                                    sum(looks_like_triage(p) for p in rpreds) / len(rpreds), 4),
                                "regression_outputs": rpreds})
        print(key, scores)
    report.write_json(summary, "replay.json", results_dir=RESULTS)


# --- sweep (B4) --------------------------------------------------------------

def cmd_sweep(args) -> None:
    budget = correct_budget()
    autopsy = json.loads((RESULTS / "autopsy.json").read_text(encoding="utf-8"))
    verdict = json.loads((RESULTS / "verdict.json").read_text(encoding="utf-8"))
    ft = next(r for r in verdict["comparison"] if r["run"].startswith("(c)"))
    corr = next(r for r in autopsy if r["run"] == "correct")
    rows = [{"r": 16, "run": "correct", "target": corr["target"], "format": corr["format"],
             "regression": ft["regression"], "source": "NB3/NB5 (not retrained)"}]
    for r in args.ranks:
        if r == 16:
            continue
        spec = SPECS["correct"].resolved(r)
        key = f"sweep_r{r}"
        print("=" * 70, f"\nRUN {key}: text-linear r={r} alpha={spec.alpha}, {budget} steps")
        tr = train_run(key, spec, TRAIN_ROWS, budget, {"label": f"text-linear · r={r} · LR 10x"})
        model, tok = load_adapter(ROOT / "adapters" / key)
        scores, _, _ = score(model, tok, label=key)
        del model
        generate.free_memory()
        rows.append({"r": r, "run": key, "trainable_params": tr["trainable_params"],
                     "final_loss": tr["final_loss"], "loss_curve": tr["loss_curve"],
                     "peak_vram_gb": tr["peak_vram_gb"], **scores, "source": "this script"})
        print(key, scores)
    rows.sort(key=lambda x: x["r"])
    report.write_json({"model": TIER.model_id, "placement": "text-linear", "max_steps": budget,
                       "rows": rows}, "rank_sweep.json", results_dir=RESULTS)


# --- qual -------------------------------------------------------------------

def cmd_qual(_args) -> None:
    """Per-item (b) vs (c) on the target set. NB2 freezes (b)'s aggregate but not its
    outputs; greedy decoding with the same batches regenerates them, and the mean is
    checked against the frozen number so this cannot silently drift from it."""
    frozen = json.loads((RESULTS / "baselines_frozen.json").read_text(encoding="utf-8"))
    ft_rows = {r["i"]: r for r in json.loads((RESULTS / "qualitative.json").read_text(encoding="utf-8"))}

    model, tok = generate.load_base(TIER)
    preds_b, _ = generate.generate_batch(model, tok, [r["input"] for r in TARGET],
                                         system=generate.OPTIMIZED_PROMPT, label="(b)/target")
    del model
    generate.free_memory()
    model, tok = load_adapter(ROOT / "adapters" / "correct")
    preds_c, _ = generate.generate_batch(model, tok, [r["input"] for r in TARGET],
                                         system=generate.NAIVE_PROMPT, label="(c)/target")
    del model
    generate.free_memory()

    items = []
    for i, (pb, pc, r) in enumerate(zip(preds_b, preds_c, TARGET)):
        sb, sc = ev.triage_field_accuracy(pb, r["label"]), ev.triage_field_accuracy(pc, r["label"])
        items.append({"i": i, "ticket": r["input"], "label": r["label"],
                      "b_pred": pb, "b_score": round(sb, 2), "c_pred": pc, "c_score": round(sc, 2),
                      "delta": round(sc - sb, 2)})
    mean_b = sum(x["b_score"] for x in items) / len(items)
    mean_c = sum(x["c_score"] for x in items) / len(items)
    same_c = all(abs(ft_rows[x["i"]]["ft_score"] - x["c_score"]) < 1e-6 for x in items)
    out = {"mean_b": round(mean_b, 4), "frozen_b": frozen["baseline_b"]["target"],
           "mean_c": round(mean_c, 4), "c_matches_nb5_qualitative": same_c,
           "n_ft_wins": sum(x["delta"] > 0 for x in items),
           "n_ties": sum(x["delta"] == 0 for x in items),
           "n_ft_losses": sum(x["delta"] < 0 for x in items),
           "items": items}
    report.write_json(out, "qualitative_compare.json", results_dir=RESULTS)
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("qual")
    p = sub.add_parser("replay")
    p.add_argument("--pct", type=int, nargs="+", default=[5])
    s = sub.add_parser("sweep")
    s.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 64])
    args = ap.parse_args()
    {"probe": cmd_probe, "qual": cmd_qual, "replay": cmd_replay, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
