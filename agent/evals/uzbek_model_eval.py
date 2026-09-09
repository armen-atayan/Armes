#!/usr/bin/env python3
"""Blinded Uzbek voice-agent comparison: Terra vs Gemini Flash.

Candidate outputs are randomized per scenario before they are shown to judges.
No candidate model name is included in a judge prompt.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_ENV = Path("/home/ubuntu/livekit-stack/.env")
REPORT = Path("/home/ubuntu/livekit-stack/agent/evals/uzbek_model_eval_report.json")


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


ENV = load_env(ROOT_ENV)


def chat(base: str, key: str, model: str, messages: list[dict], max_tokens: int = 700) -> dict:
    payload = {"model": model, "messages": messages}
    is_azure = ".openai.azure.com" in base
    # Azure Foundry's reasoning models only accept the default temperature;
    # the Gen2B Gemini adapter accepts an explicit low temperature.
    if not is_azure:
        payload["temperature"] = 0.2
    # Azure Foundry's v1 endpoint accepts max_completion_tokens; the Gen2B
    # gateway's Gemini adapter accepts max_tokens.
    payload["max_completion_tokens" if is_azure else "max_tokens"] = max_tokens
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
        choice = body["choices"][0]
        return {
            "ok": True,
            "text": choice.get("message", {}).get("content") or "",
            "returned_model": body.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "latency_s": round(time.perf_counter() - started, 2),
        }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            error = json.loads(raw).get("error", {})
        except json.JSONDecodeError:
            error = {"message": raw[:500]}
        return {"ok": False, "status": exc.code, "error": error, "latency_s": round(time.perf_counter() - started, 2)}


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
    "terra": {
        "base": ENV["GEN2B_LLM_BASE"],
        "key": ENV["GEN2B_LLM_KEY"],
        "model": "gpt-5.6-terra",
    },
    "gemini_flash": {
        "base": ENV["GEN2B_BASE"],
        "key": ENV["GEN2B_KEY"],
        "model": "gemini/gemini-2.5-flash",
    },
}

JUDGES = {
    "gemini_3_1_pro": {
        "base": ENV["GEN2B_BASE"],
        "key": ENV["GEN2B_KEY"],
        "model": "gemini/gemini-3.1-pro-preview",
    },
    # Azure currently returns DeploymentNotFound for this resource/model.
    "sol": {
        "base": ENV["GEN2B_LLM_BASE"],
        "key": ENV["GEN2B_LLM_KEY"],
        "model": "gpt-5.6-sol",
    },
}


def parse_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(match.group(0) if match else text)


def main() -> None:
    report: dict = {"scenarios": SCENARIOS, "candidates": {}, "judgments": {}, "judge_availability": {}}
    for sid in [s["id"] for s in SCENARIOS]:
        scenario = next(s for s in SCENARIOS if s["id"] == sid)
        report["candidates"][sid] = {}
        for label, config in CANDIDATES.items():
            result = chat(config["base"], config["key"], config["model"], [{"role": "system", "content": SYSTEM}, *scenario["history"]], 700)
            report["candidates"][sid][label] = result

    sol_probe = chat(JUDGES["sol"]["base"], JUDGES["sol"]["key"], JUDGES["sol"]["model"], [{"role": "user", "content": "Reply exactly: OK"}], 64)
    report["judge_availability"]["sol"] = sol_probe
    report["judge_availability"]["gemini_3_1_pro"] = {"ok": True, "model": JUDGES["gemini_3_1_pro"]["model"]}

    # Fixed seed makes A/B blinding reproducible while remaining model-name-free to judge.
    rng = random.Random(20260901)
    rubric = """Sen o‘zbekcha telefon-sotuv agenti sifatini baholovchi mustaqil hakamsan. Ikki anonim javobni A va B sifatida bahola. Mezonga amal qil: tabiiy o‘zbek tili, aynan shu suhbatga moslik, qisqalik/telefon uslubi, bir turda bitta savol, faktlarni to‘qimaslik, kalendar bo‘yicha yolg‘on va’da bermaslik. Natijani faqat JSON qilib qaytar: {\"winner\":\"A\"|\"B\"|\"tie\",\"scores\":{\"A\":0-10,\"B\":0-10},\"reason_uz\":\"qisqa o‘zbekcha sabab\"}."""
    for scenario in SCENARIOS:
        outputs = report["candidates"][scenario["id"]]
        order = ["terra", "gemini_flash"]
        rng.shuffle(order)
        mapping = {"A": order[0], "B": order[1]}
        prompt = f"Vaziyat: {scenario['title']}\n\nA javob:\n{outputs[mapping['A']]['text']}\n\nB javob:\n{outputs[mapping['B']]['text']}"
        raw = chat(
            JUDGES["gemini_3_1_pro"]["base"],
            JUDGES["gemini_3_1_pro"]["key"],
            JUDGES["gemini_3_1_pro"]["model"],
            [{"role": "system", "content": rubric}, {"role": "user", "content": prompt}],
            2000,
        )
        item = {"mapping": mapping, "raw": raw}
        if raw.get("ok"):
            try:
                item["verdict"] = parse_json(raw["text"])
            except Exception as exc:
                item["parse_error"] = str(exc)
        report["judgments"][scenario["id"]] = item

    wins = {"terra": 0, "gemini_flash": 0, "tie": 0}
    for item in report["judgments"].values():
        verdict = item.get("verdict", {})
        winner = verdict.get("winner")
        if winner in ("A", "B"):
            wins[item["mapping"][winner]] += 1
        else:
            wins["tie"] += 1
    report["summary"] = {"gemini_3_1_pro_wins": wins, "sol_judge_status": sol_probe}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
