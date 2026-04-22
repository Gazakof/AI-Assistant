# models/qcm_generator.py

import random
import json
import re
from transformers import pipeline

# ✅ modèle adapté
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# =========================
# 🔹 EXTRACTION JSON
# =========================
def extract_json(text):
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)

    return json.loads(text)


# =========================
# 🔹 SPLIT TEXTE (IMPORTANT)
# =========================
def split_text(text, chunk_size=400):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# =========================
# 🔹 QCM IA
# =========================
def generate_qcm_dl(text, num_questions=5):
    try:
        text = text[:1200]

        prompt = f"""
You are a university professor.

Create {num_questions} high-quality multiple choice questions from the text.

Rules:
- Questions must test understanding (not just copy text)
- Avoid obvious answers
- Include tricky distractors
- 4 options exactly
- Only 1 correct answer

Return ONLY JSON:

[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }}
]

Text:
{text}
"""

        result = generator(
            prompt,
            max_new_tokens=300,
            do_sample=False
        )

        output = result[0]["generated_text"]
        qcm = extract_json(output)

        cleaned = []
        for q in qcm:
            if (
                isinstance(q, dict)
                and "question" in q
                and "options" in q
                and "answer" in q
                and len(q["options"]) == 4
                and q["answer"] in q["options"]
            ):
                cleaned.append({
                    "question": q["question"],
                    "options": q["options"],
                    "answer": q["answer"]
                })

        if cleaned:
            return cleaned

        return generate_qcm_basic(text)

    except Exception as e:
        print("⚠️ QCM DL failed:", e)
        return generate_qcm_basic(text)


# =========================
# 🔹 FALLBACK SIMPLE
# =========================
def generate_qcm_basic(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    questions = []

    for sent in sentences[:5]:
        words = sent.split()
        if len(words) > 5:
            answer = words[min(2, len(words)-1)]
            question = sent.replace(answer, "_____")

            options = list({answer, *random.sample(words, min(3, len(words)))})
            while len(options) < 4:
                options.append(random.choice(words))
            random.shuffle(options)
            options = options[:4]

            questions.append({
                "question": question,
                "options": options,
                "answer": answer
            })

    return questions


# =========================
# 🔥 FONCTION PRINCIPALE (BOOST)
# =========================
def generate_qcm(text):
    chunks = split_text(text)

    all_questions = []

    for chunk in chunks:
        qcm = generate_qcm_dl(chunk, num_questions=5)
        all_questions.extend(qcm)

    # 🔀 Mélanger les questions
    random.shuffle(all_questions)

    # 🎯 Limite finale (modifiable)
    return all_questions[:10]