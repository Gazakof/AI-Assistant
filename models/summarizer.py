import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# =========================
# 🚀 CONFIGURATION DU MODÈLE
# =========================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("🔄 Chargement du Summarizer (Qwen 4-bit)...")

try:
    # On utilise la même configuration que pour le chatbot pour réutiliser la VRAM
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Utilisation de la tâche "text-generation" (plus flexible que "summarization" pour les modèles Instruct)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    print("✅ Summarizer prêt.")
except Exception as e:
    print(f"⚠️ Erreur chargement Summarizer : {e}")
    generator = None


# =========================
# 📝 LOGIQUE DE RÉSUMÉ
# =========================

def summarize(text):
    if not text or len(text.strip()) < 100:
        return text

    if not generator:
        return "Erreur : Modèle non chargé."

    try:
        # Nettoyage rapide
        clean_text = re.sub(r'\s+', ' ', text).strip()

        # Limitation de la longueur d'entrée (Qwen gère bien plus, mais on reste efficace)
        # On prend les 4000 premiers caractères pour la synthèse
        truncated_text = clean_text[:4000]

        # Construction du Prompt pour Qwen
        messages = [
            {
                "role": "system",
                "content": "Tu es un expert en synthèse de documents. Ton but est de rédiger un résumé clair, structuré et fidèle au texte original en français."
            },
            {
                "role": "user",
                "content": f"Fais un résumé synthétique du texte suivant en environ 3 à 5 phrases :\n\n{truncated_text}"
            }
        ]

        prompt = generator.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Génération
        outputs = generator(
            prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.3,  # On reste factuel
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=generator.tokenizer.eos_token_id
        )

        # Extraction de la réponse
        full_text = outputs[0]["generated_text"]
        summary = full_text.split("<|im_start|>assistant")[-1].strip()

        # Nettoyage final des sauts de ligne excessifs
        summary = re.sub(r'\n{3,}', '\n\n', summary)

        return summary

    except Exception as e:
        print(f"❌ Erreur lors du résumé : {e}")
        # Petit fallback manuel si Qwen échoue
        return text[:300] + "..."


# Test rapide
if __name__ == "__main__":
    test_text = """L'intelligence artificielle est un domaine en pleine expansion. 
    De nombreux chercheurs travaillent sur des modèles de langage de plus en plus performants. 
    Ces modèles, comme Qwen ou GPT, permettent de traiter des volumes massifs de données 
    pour extraire des informations pertinentes ou générer du contenu de manière autonome."""
    print(summarize(test_text))