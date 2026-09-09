#!/usr/bin/env python3
"""Production-path A/B: Gemini 2.5 Flash (none) vs GPT-5.6 Luna (none)."""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "evals" / "gemini25_vs_luna56_report.json"


def load_env(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"').strip("'")
    return values


ENV = load_env(ROOT.parent / ".env")
SYSTEM = """Sen Bito ERP platformasining CEOsi Uchqunsan. Ovozli qo‘ng‘iroqda faqat tabiiy o‘zbek tilida gapir. Juda qisqa (1–2 jumla) javob ber, bir turda faqat bitta savol ber, Markdown ishlatma. Hech qanday tasdiqlanmagan fakt yoki narxni o‘ylab topma. Bito bo‘yicha tasdiqlangan ma’lumotlar: ERP savdo, CRM, ombor, logistika, HR, ishlab chiqarish va analitikaga yordam beradi; Start tarifi saytda 300 000 so‘mdan boshlanadi; Individual tarifi ehtiyoj o‘rganilgach shakllanadi; 12 oylik obunaga 2 oy bonus taklifi bor. Start tarifining davri, foydalanuvchi yoki modul limiti tasdiqlanmagan. Demo/konsultatsiyaga real kalendar tasdig‘isiz yozib qo‘yilgan deb da’vo qilma."""
SCENARIOS = [
    {"id": "discovery", "title": "Tasdiqlangan suhbatdosh va birinchi discovery savoli", "history": [
        {"role": "assistant", "content": "Assalomu alaykum, bu Diyorbekmi?"},
        {"role": "user", "content": "Ha, men Diyorbekman."},
        {"role": "assistant", "content": "Men Uchqun, Bito ERP platformasining CEOsi bo‘laman. Biznes jarayonlarini avtomatlashtirish bo‘yicha qisqa suhbat uchun qo‘ng‘iroq qildim."},
        {"role": "user", "content": "Mayli, eshitaman."},
    ]},
    {"id": "russian_caller", "title": "Ruscha gapirgan mijozga o‘zbekcha, bitta savolli javob", "history": [
        {"role": "assistant", "content": "Sizning biznesingiz qaysi yo‘nalishda ishlaydi?"},
        {"role": "user", "content": "У нас три магазина одежды и один склад, учёт ведём в Excel."},
    ]},
    {"id": "pricing_truth", "title": "Tasdiqlanmagan tarif detallarini to‘qimaslik", "history": [
        {"role": "user", "content": "Start tarifi 300 ming so‘m ekan. Shu narx oylikmi, nechta foydalanuvchi va qaysi modullar kiradi?"},
    ]},
    {"id": "demo_booking", "title": "Demo uchun keyingi qadam, soxta kalendar tasdig‘isiz", "history": [
        {"role": "user", "content": "CRM va ombor bizga qiziq. Demoni payshanba kuni soat uchda qilaylik."},
    ]},
]
CANDIDATES = {
    "gemini_2_5_flash_none": {
        "base": ENV["GEN2B_BASE"].rstrip("/"), "key": ENV["GEN2B_KEY"],
        "model": "gemini/gemini-2.5-flash", "token_key": "max_tokens", "temperature": 0.2,
    },
    "gpt_5_6_luna_none": {
        "base": "https://armes-resource.services.ai.azure.com/api/projects/armes/openai/v1",
        "key": ENV["GEN2B_LLM_KEY"], "model": "gpt-5.6-luna", "token_key": "max_completion_tokens",
    },
}


def stream_chat(cfg: dict, messages: list[dict], limit: int = 1200) -> dict:
    payload = {
        "model": cfg["model"], "messages": messages, "reasoning_effort": "none",
        cfg["token_key"]: limit, "stream": True, "stream_options": {"include_usage": True},
    }
    if "temperature" in cfg:
        payload["temperature"] = cfg["temperature"]
    headers = {"Authorization": "Bearer " + cfg["key"], "Content-Type": "application/json"}
    started = time.perf_counter(); first = None; parts = []; usage = None; finish = None
    with requests.post(cfg["base"] + "/chat/completions", headers=headers, json=payload, stream=True, timeout=180) as response:
        if not response.ok:
            return {"ok": False, "status": response.status_code, "error": response.text[:1000], "total_latency_s": round(time.perf_counter() - started, 3)}
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if not choices:
                continue
            finish = choices[0].get("finish_reason") or finish
            text = (choices[0].get("delta") or {}).get("content") or ""
            if text:
                first = first or time.perf_counter(); parts.append(text)
    ended = time.perf_counter()
    return {"ok": True, "status": 200, "text": "".join(parts).strip(), "ttft_s": round(first-started, 3) if first else None, "total_latency_s": round(ended-started, 3), "finish_reason": finish, "usage": usage}


def judge(prompt: str) -> dict:
    cfg = {"base": ENV["GEN2B_BASE"].rstrip("/"), "key": ENV["GEN2B_KEY"], "model": "gemini/gemini-3.1-pro-preview"}
    payload = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000, "temperature": 0.1}
    started = time.perf_counter()
    response = requests.post(cfg["base"] + "/chat/completions", headers={"Authorization": "Bearer " + cfg["key"]}, json=payload, timeout=180)
    if not response.ok:
        return {"ok": False, "status": response.status_code, "error": response.text[:1000]}
    body = response.json()
    return {"ok": True, "text": (body["choices"][0].get("message") or {}).get("content") or "", "latency_s": round(time.perf_counter()-started, 3)}


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text.strip(), flags=re.S)
    return json.loads(match.group(0) if match else text)


def main() -> None:
    report = {"candidates": {k: {kk: vv for kk, vv in v.items() if kk != "key"} for k, v in CANDIDATES.items()}, "scenarios": {}, "judgments": {}}
    for scenario in SCENARIOS:
        outputs = {label: stream_chat(cfg, [{"role": "system", "content": SYSTEM}, *scenario["history"]]) for label, cfg in CANDIDATES.items()}
        report["scenarios"][scenario["id"]] = {"title": scenario["title"], "outputs": outputs}

    rubric = """Sen o‘zbekcha telefon-sotuv agenti sifatini baholovchi mustaqil hakamsan. Ikki anonim javobni A va B sifatida bahola: tabiiy o‘zbek tili, vaziyatga moslik, qisqalik, bitta savol, faktlarni to‘qimaslik va soxta kalendar va’dasidan qochish. Faqat JSON qaytar: {\"winner\":\"A\"|\"B\"|\"tie\",\"scores\":{\"A\":0-10,\"B\":0-10},\"reason_uz\":\"qisqa sabab\"}."""
    rng = random.Random(20260903); wins = {**{k: 0 for k in CANDIDATES}, "tie": 0}
    for scenario in SCENARIOS:
        order = list(CANDIDATES); rng.shuffle(order); mapping = {"A": order[0], "B": order[1]}
        outputs = report["scenarios"][scenario["id"]]["outputs"]
        raw = judge(rubric + f"\n\nVaziyat: {scenario['title']}\n\nA:\n{outputs[mapping['A']]['text']}\n\nB:\n{outputs[mapping['B']]['text']}")
        item = {"mapping": mapping, "raw": raw}
        try:
            verdict = parse_json(raw["text"]); item["verdict"] = verdict
            winner = verdict.get("winner"); wins[mapping[winner] if winner in mapping else "tie"] += 1
        except Exception as exc:
            item["parse_error"] = str(exc); wins["tie"] += 1
        report["judgments"][scenario["id"]] = item

    latency = {}
    for label in CANDIDATES:
        rows = [report["scenarios"][s["id"]]["outputs"][label] for s in SCENARIOS]
        ttft = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
        total = [r["total_latency_s"] for r in rows if r.get("ok")]
        reasoning = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for r in rows]
        latency[label] = {"mean_ttft_s": round(sum(ttft)/len(ttft), 3), "mean_total_latency_s": round(sum(total)/len(total), 3), "reasoning_tokens_total": sum(reasoning), "successful": sum(bool(r.get("text")) for r in rows)}
    report["summary"] = {"wins": wins, "latency": latency}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
