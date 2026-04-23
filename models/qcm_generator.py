import random
import json
import re
import logging
from collections import deque
from typing import TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# =========================
# 📝 LOGGING
# =========================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================
# 🚀 CONFIGURATION
# =========================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = "D:/Projet/AI/models"
TOTAL_QUESTIONS = 8
MAX_WORDS = 350   # légèrement plus pour donner plus de contexte au modèle
MAX_RETRIES = 2
MAX_NEW_TOKENS = 512   # ↓ depuis 800 — suffisant pour 8 questions compactes

# =========================
# ⚡ REGEX COMPILÉS (module-level, O(1) lookup)
# =========================
_RE_FENCE      = re.compile(r"```(?:json)?")
_RE_CTRL       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_RE_ARRAY      = re.compile(r"\[.*\]", re.DOTALL)
_RE_OBJ        = re.compile(r"\{.*\}",  re.DOTALL)
_RE_SPLIT_SENT = re.compile(r"(?<=[.!?])\s+")

# =========================
# 🗂️ TYPES & MAPPING DE CLÉS
# =========================
class Question(TypedDict):
    question: str
    options:  list[str]
    answer:   str

# Accepte clés longues OU courtes (q/o/a) générées par le modèle
_KEY_MAP: dict[str, tuple[str, ...]] = {
    "question": ("question", "q"),
    "options":  ("options",  "o"),
    "answer":   ("answer",   "a"),
}

def _resolve(raw: dict, key: str):
    """Résout une clé depuis ses alias possibles."""
    for alias in _KEY_MAP[key]:
        if alias in raw:
            return raw[alias]
    return [] if key == "options" else ""


# =========================
# 🔄 SINGLETON MODÈLE (lazy load)
# =========================
_generator = None


def get_generator():
    """Charge le modèle une seule fois ; réutilise l'instance ensuite."""
    global _generator
    if _generator is not None:
        return _generator

    logger.info("🔄 Chargement du modèle (%s)…", MODEL_ID)

    # 🔥 Config quantization (comme chat.py)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir = CACHE_DIR)

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            # torch_dtype=torch.float16,
            # attn_implementation="sdpa",   # Flash-Attention 2 si dispo, SDPA sinon
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True
        )
        
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("⚡ torch.compile activé (reduce-overhead).")
        except Exception as e:
            logger.warning("⚠️ torch.compile indisponible : %s", e)

        logger.info("✅ Modèle chargé sur GPU.")

    else:
        logger.warning("⚠️ GPU non disponible — mode CPU.")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True
        )
    

    _generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,   # ← ne retourne QUE la partie générée (plus rapide)
    )
    return _generator


# =========================
# 🔹 EXTRACTION JSON (robuste)
# =========================
def extract_json(text: str) -> list[dict]:
    """Extrait une liste JSON depuis la sortie brute du modèle."""
    text = _RE_FENCE.sub("", text).strip()
    text = _RE_CTRL.sub("", text)

    m = _RE_ARRAY.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    m = _RE_OBJ.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return [obj] if isinstance(obj, dict) else []
        except (json.JSONDecodeError, ValueError):
            pass

    logger.debug("Aucun JSON exploitable trouvé dans la sortie.")
    return []


# =========================
# ✅ VALIDATION RENFORCÉE
# =========================
def validate_question(q: dict) -> bool:
    """
    Vérifie :
      - présence des clés obligatoires
      - exactement 4 options, toutes non-vides
      - aucun doublon (insensible à la casse)
      - la réponse figure dans les options
      - la question est suffisamment longue (évite les questions tronquées)
    """
    if not isinstance(q, dict):
        return False
    if not {"question", "options", "answer"}.issubset(q):
        return False

    quest = q["question"].strip()
    if len(quest) < 15:           # question trop courte = tronquée ou vide
        return False

    options = q.get("options", [])
    if len(options) != 4:
        return False
    if any(not isinstance(o, str) or not o.strip() for o in options):
        return False

    # Unicité O(n) via set
    normalized = {o.strip().lower() for o in options}
    if len(normalized) != 4:
        return False

    answer_low = q["answer"].strip().lower()
    if not answer_low or answer_low not in normalized:
        return False

    return True


# =========================
# 🔧 DÉDUPLICATION DES OPTIONS (O(n))
# =========================
def deduplicate_options(q: dict, word_pool: list[str]) -> dict:
    """Remplace les options en doublon par des mots tirés du texte source."""
    options     = [o.strip() for o in q["options"]]
    answer_low  = q["answer"].strip().lower()

    pool_map: dict[str, str] = {w.lower(): w for w in word_pool}
    seen:    set[str]        = set()
    fixed:   list[str]       = []

    for opt in options:
        opt_low = opt.lower()
        if opt_low not in seen:
            fixed.append(opt)
            seen.add(opt_low)
        else:
            # Cherche un remplaçant dans le pool
            replacement = next(
                (pool_map.pop(w) for w in list(pool_map)
                 if w not in seen and w != answer_low),
                f"Autre option {len(fixed) + 1}"
            )
            fixed.append(replacement)
            seen.add(fixed[-1].lower())

    q["options"] = fixed
    return q


# =========================
# 🔥 PROMPT HAUTE QUALITÉ
# =========================
# Types de questions imposés au modèle pour forcer la variété
_QUESTION_TYPES = (
    "définition (Qu'est-ce que…)",
    "cause / raison (Pourquoi… / Quelle est la cause…)",
    "conséquence / effet (Quel est l'effet de… / Qu'entraîne…)",
    "identification (Lequel de ces éléments… / Quel est le nom de…)",
    "comparaison (Quelle différence entre… / Lequel est le plus…)",
    "rôle / fonction (Quel est le rôle de… / À quoi sert…)",
    "chronologie (Quand… / Dans quel ordre…)",
    "exemple (Quel exemple illustre…)",
)

# Un seul exemple few-shot très compact pour guider sans gonfler le prompt
_FEW_SHOT_EXAMPLE = (
    '[{"q":"Quel est le rôle principal du Soleil dans le système solaire ?","o":'
    '["Centre gravitationnel","Unique source de lumière","La plus grande planète","Le corps le plus froid"],'
    '"a":"Centre gravitationnel"}]'
)


def build_prompt(text: str, n: int) -> str:
    """
    Construit un prompt orienté qualité :
      - types de questions variés imposés
      - distracteurs plausibles exigés
      - exemple few-shot pour ancrer le format
    """
    gen = get_generator()

    # On limite le nombre de types affichés pour ne pas allonger inutilement le prompt
    shown_types = "\n".join(f"• {t}" for t in _QUESTION_TYPES[:min(n, len(_QUESTION_TYPES))])

    system_msg = (
        "Tu es un expert en création de QCM éducatifs francophones. "
        "Réponds UNIQUEMENT avec du JSON valide, sans explication ni balise Markdown. "
        "Règles impératives :\n"
        "1. Génère des questions qui testent la COMPRÉHENSION, pas la mémorisation littérale.\n"
        "2. Varie les types de questions selon cette liste :\n"
        f"{shown_types}\n"
        "3. Chaque option doit être plausible, unique et de longueur comparable.\n"
        "4. La réponse correcte doit être l'une des 4 options, mot pour mot.\n"
        f"5. Format strict (clés q/o/a) : {_FEW_SHOT_EXAMPLE}"
    )

    user_msg = (
        f"Génère {n} QCM de haute qualité en français sur le texte suivant.\n"
        f"Texte : {text}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

    return gen.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =========================
# 🆘 FALLBACK QUALITATIF
# =========================
def generate_qcm_fallback(text: str) -> list[dict]:
    """
    Fallback basé sur des questions de compréhension réelles (pas fill-in-the-blank),
    avec distracteurs issus du texte pour rester plausibles.
    """
    sentences = [
        s.strip()
        for s in _RE_SPLIT_SENT.split(text)
        if len(s.strip().split()) >= 8
    ]

    # Pool de noms/groupes nominaux (mots >4 chars, uniques)
    all_words: list[str] = list({
        w.strip(",.;:()\"\\'«»")
        for w in text.split()
        if len(w.strip(",.;:()\"\\'«»")) > 4
    })
    random.shuffle(all_words)
    word_deque = deque(all_words)

    templates = [
        "De quoi parle principalement ce passage : « {sent} » ?",
        "Quelle affirmation correspond au texte : « {sent} » ?",
        "Quelle est l'idée principale exprimée dans : « {sent} » ?",
    ]

    fallback: list[dict] = []
    generic = deque([
        "Information non mentionnée",
        "Aucune de ces réponses",
        "Donnée absente du texte",
        "Hors-sujet du texte",
    ])

    for sent in sentences[:TOTAL_QUESTIONS]:
        words    = sent.split()
        answer   = words[len(words) // 2].strip(",.;:()\"\\'«»")
        ans_low  = answer.lower()

        distractors: list[str] = []
        for candidate in list(word_deque):
            if candidate.lower() != ans_low and candidate not in distractors:
                distractors.append(candidate)
            if len(distractors) == 3:
                break

        while len(distractors) < 3:
            distractors.append(generic[0] if generic else f"Option {len(distractors)+1}")
            if generic:
                generic.rotate(-1)

        question_text = random.choice(templates).format(sent=sent[:80])
        options = [answer] + distractors
        random.shuffle(options)

        fallback.append({"question": question_text, "options": options, "answer": answer})

    return fallback


# =========================
# 🚀 GÉNÉRATION PRINCIPALE
# =========================
def generate_qcm(text: str) -> list[Question]:
    """
    Génère `TOTAL_QUESTIONS` QCM à partir d'un texte.

    Stratégie :
      1. Tentative principale (toutes les questions d'un coup).
      2. Relance ciblée pour les questions manquantes (MAX_RETRIES).
      3. Fallback si toujours insuffisant.
    """
    gen = get_generator()
    if not gen:
        return generate_qcm_fallback(text)

    clean_text = " ".join(text.split()[:MAX_WORDS])
    word_pool  = list({
        w.strip(",.;:()\"\\'«»")
        for w in clean_text.split()
        if len(w) > 4
    })
    random.shuffle(word_pool)

    questions:      list[Question] = []
    seen_questions: set[str]       = set()

    for attempt in range(MAX_RETRIES):
        if len(questions) >= TOTAL_QUESTIONS:
            break

        remaining = TOTAL_QUESTIONS - len(questions)
        logger.info(
            "🔁 Tentative %d/%d — génération de %d question(s)…",
            attempt + 1, MAX_RETRIES, remaining,
        )

        try:
            prompt = build_prompt(clean_text, remaining)
            result = gen(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy = déterministe + plus rapide
                # pas de repetition_penalty : coûteux et inutile avec do_sample=False
                pad_token_id=gen.tokenizer.eos_token_id,
            )

            # Avec return_full_text=False, le texte généré est directement dans result[0]
            raw_output = result[0]["generated_text"]
            extracted  = extract_json(raw_output)

            for raw_q in extracted:
                if not isinstance(raw_q, dict):
                    continue

                q: Question = {
                    "question": str(_resolve(raw_q, "question")).strip(),
                    "options":  [str(o) for o in _resolve(raw_q, "options")],
                    "answer":   str(_resolve(raw_q, "answer")).strip(),
                }

                if not q["question"]:
                    continue

                q_key = q["question"].lower()
                if q_key in seen_questions:
                    continue

                q = deduplicate_options(q, word_pool)

                if validate_question(q):
                    questions.append(q)
                    seen_questions.add(q_key)
                    logger.info("✅ Question valide (%d/%d)", len(questions), TOTAL_QUESTIONS)
                else:
                    logger.warning(
                        "⚠️ Rejetée (tentative %d) : %s",
                        attempt + 1, q.get("question", "")[:70],
                    )

        except Exception as e:
            logger.error("⚠️ Erreur tentative %d : %s", attempt + 1, e)

    # Fallback si insuffisant
    if len(questions) < TOTAL_QUESTIONS:
        needed = TOTAL_QUESTIONS - len(questions)
        logger.info("🆘 Fallback pour %d question(s) manquante(s)…", needed)
        questions.extend(generate_qcm_fallback(clean_text)[:needed])

    return questions[:TOTAL_QUESTIONS]


# =========================
# 🧪 TEST
# =========================
if __name__ == "__main__":
    sample = (
        "La photosynthèse est le processus par lequel les plantes, les algues et certaines "
        "bactéries convertissent l'énergie lumineuse en énergie chimique stockée sous forme "
        "de glucose. Ce phénomène se produit principalement dans les chloroplastes, grâce à "
        "la chlorophylle, le pigment vert responsable de l'absorption de la lumière. "
        "La réaction globale consomme du dioxyde de carbone et de l'eau, et libère de "
        "l'oxygène comme sous-produit, jouant ainsi un rôle crucial dans le cycle du carbone "
        "et dans le maintien de la composition atmosphérique terrestre."
    )

    results = generate_qcm(sample)
    for i, q in enumerate(results, 1):
        print(f"\nQ{i}: {q['question']}")
        for opt in q["options"]:
            marker = "✅" if opt.strip().lower() == q["answer"].strip().lower() else "   "
            print(f"  {marker} {opt}")