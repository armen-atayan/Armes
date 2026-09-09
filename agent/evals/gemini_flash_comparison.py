#!/usr/bin/env python3
"""Compare current Gemini 2.5 Flash/no-thinking with latest available Gemini 3.7 Flash/low."""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
ENV_PATH = ROOT.parent / ".env"
REPORT_PATH = ROOT / "evals" / "gemini_25_vs_37_flash_report.json"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"').strip("'")
    return values


ENV = load_env(ENV_PATH)
BASE = ENV["GEN2B_BASE"].rstrip("/")
KEY = ENV["GEN2B_KEY"]
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

SYSTEM = """Sen Bito ERP platformasining CEOsi Uchqunsan. Ovozli qo‘ng‘iroqda faqat tabiiy o‘zbek tilida gapir. Juda qisqa (1–2 jumla) javob ber, bir turda faqat bitta savol ber, Markdown ishlatma. Hech qanday tasdiqlanmagan fakt yoki narxni o‘ylab topma. Bito bo‘yicha tasdiqlangan ma’lumotlar: ERP savdo, CRM, ombor, logistika, HR, ishlab chiqarish va analitikaga yordam beradi; Start tarifi saytda 300 000 so‘mdan boshlanadi; Individual tarifi ehtiyoj o‘rganilgach shakllanadi; 12 oylik obunaga 2 oy bonus taklifi bor. Start tarifining davri, foydalanuvchi yoki modul limiti tasdiqlanmagan. Demo/konsultatsiyaga real kalendar tasdig‘isiz yozib qo‘yilgan deb da’vo qilma."""

SCENARIOS = [
    {
        "id": "discovery",
        "title": "Tasdiqlangan suhbatdosh va birinchi discovery savoli",
        "history": [
            {"role": "assistant", "content": "Assalomu alaykum, bu Diyorbekmi?"},
            {"role": "user", "content": "Ha, men Diyorbekman."},
            {"role": "assistant", "content": "Men Uchqun, Bito ERP platformasining CEOsi bo‘laman. Biznes jarayonlarini avtomatlashtirish bo‘yicha qisqa suhbat uchun qo‘ng‘iroq qildim."},
            {"role": "user", "content": "Mayli, eshitaman."},
        ],
    },
    {
        "id": "russian_caller",
        "title": "Ruscha gapirgan mijozga o‘zbekcha, bitta savolli javob",
        "history": [
            {"role": "assistant", "content": "Sizning biznesingiz qaysi yo‘nalishda ishlaydi?"},
            {"role": "user", "content": "У нас три магазина одежды и один склад, учёт ведём в Excel."},
        ],
    },
    {
        "id": "pricing_truth",
        "title": "Tasdiqlanmagan tarif detallarini to‘qimaslik",
        "history": [
            {"role": "user", "content": "Start tarifi 300 ming so‘m ekan. Shu narx oylikmi, nechta foydalanuvchi va qaysi modullar kiradi?"},
        ],
    },
    {
        "id": "demo_booking",
        "title": "Demo uchun keyingi qadam, soxta kalendar tasdig‘isiz",
        "history": [
            {"role": "user", "content": "CRM va ombor bizga qiziq. Demoni payshanba kuni soat uchda qilaylik."},
        ],
    },
]

CANDIDATES = {
    "gemini_2_5_flash_none": {"model": "gemini/gemini-2.5-flash", "reasoning_effort": "none"},
    "gemini_3_7_flash_low": {"model": "gemini-3.7-flash", "reasoning_effort": "low"},
}


def stream_chat(model: str, effort: str, messages: list[dict], max_tokens: int = 500) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_content_at = None
    content_parts: list[str] = []
    usage = None
    finish_reason = None
    with requests.post(BASE + "/chat/completions", headers=HEADERS, json=payload, stream=True, timeout=180) as response:
        status = response.status_code
        if not response.ok:
            return {"ok": False, "status": status, "error": response.text[:1000], "total_latency_s": round(time.perf_counter() - started, 3)}
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            text = (choice.get("delta") or {}).get("content") or ""
            if text:
                if first_content_at is None:
                    first_content_at = time.perf_counter()
                content_parts.append(text)
    ended = time.perf_counter()
    return {
        "ok": True,
        "status": status,
        "text": "".join(content_parts).strip(),
        "ttft_s": round(first_content_at - started, 3) if first_content_at else None,
        "total_latency_s": round(ended - started, 3),
        "finish_reason": finish_reason,
        "usage": usage,
    }


def nonstream_chat(model: str, messages: list[dict], max_tokens: int = 2000) -> dict:
    payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": max_tokens}
    started = time.perf_counter()
    response = requests.post(BASE + "/chat/completions", headers=HEADERS, json=payload, timeout=180)
    elapsed = round(time.perf_counter() - started, 3)
    if not response.ok:
        return {"ok": False, "status": response.status_code, "error": response.text[:1000], "latency_s": elapsed}
    body = response.json()
    return {"ok": True, "text": (body["choices"][0].get("message") or {}).get("content") or "", "latency_s": elapsed, "usage": body.get("usage")}


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text.strip(), flags=re.S)
    return json.loads(match.group(0) if match else text)


def main() -> None:
    report: dict = {"candidates": CANDIDATES, "scenarios": {}, "judgments": {}}
    for scenario in SCENARIOS:
        outputs = {}
        for label, cfg in CANDIDATES.items():
            outputs[label] = stream_chat(cfg["model"], cfg["reasoning_effort"], [{"role": "system", "content": SYSTEM}, *scenario["history"]])
        report["scenarios"][scenario["id"]] = {"title": scenario["title"], "outputs": outputs}

    rubric = """Sen o‘zbekcha telefon-sotuv agenti sifatini baholovchi mustaqil hakamsan. Ikki anonim javobni A va B sifatida bahola. Mezonga amal qil: tabiiy o‘zbek tili, aynan shu suhbatga moslik, qisqalik/telefon uslubi, bir turda bitta savol, faktlarni to‘qimaslik, kalendar bo‘yicha yolg‘on va’da bermaslik. Natijani faqat JSON qilib qaytar: {\"winner\":\"A\"|\"B\"|\"tie\",\"scores\":{\"A\":0-10,\"B\":0-10},\"reason_uz\":\"qisqa o‘zbekcha sabab\"}."""
    rng = random.Random(20260903)
    wins = {name: 0 for name in CANDIDATES}
    wins["tie"] = 0
    for scenario in SCENARIOS:
        order = list(CANDIDATES)
        rng.shuffle(order)
        mapping = {"A": order[0], "B": order[1]}
        outputs = report["scenarios"][scenario["id"]]["outputs"]
        prompt = f"Vaziyat: {scenario['title']}\n\nA javob:\n{outputs[mapping['A']]['text']}\n\nB javob:\n{outputs[mapping['B']]['text']}"
        raw = nonstream_chat("gemini/gemini-3.1-pro-preview", [{"role": "system", "content": rubric}, {"role": "user", "content": prompt}])
        judgment = {"mapping": mapping, "raw": raw}
        if raw.get("ok"):
            try:
                verdict = parse_json(raw["text"])
                judgment["verdict"] = verdict
                winner = verdict.get("winner")
                if winner in mapping:
                    wins[mapping[winner]] += 1
                else:
                    wins["tie"] += 1
            except Exception as exc:
                judgment["parse_error"] = str(exc)
                wins["tie"] += 1
        else:
            wins["tie"] += 1
        report["judgments"][scenario["id"]] = judgment

    latency = {}
    for label in CANDIDATES:
        rows = [report["scenarios"][s["id"]]["outputs"][label] for s in SCENARIOS]
        ttfts = [row["ttft_s"] for row in rows if row.get("ttft_s") is not None]
        totals = [row["total_latency_s"] for row in rows if row.get("ok")]
        latency[label] = {
            "mean_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            "mean_total_latency_s": round(sum(totals) / len(totals), 3) if totals else None,
            "successful": sum(bool(row.get("ok") and row.get("text")) for row in rows),
        }
    report["summary"] = {"wins": wins, "latency": latency}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
