import torch, re, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────────
# 🚀 CONFIGURATION DES MODÈLES
# ──────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "D:/Projet/AI/models"

# Configuration de quantification pour Qwen (8GB VRAM friendly)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("🔄 Chargement des modèles (Embedding + Qwen 4-bit)...")

# Modèles de recherche (CPU/GPU auto via sentence-transformers)
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder = CACHE_DIR)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', cache_folder = CACHE_DIR)

# Modèle de génération (Qwen 2.5)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    cache_dir = CACHE_DIR
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config = quant_config,
    device_map = "auto",
    trust_remote_code = True,
    cache_dir = CACHE_DIR,
    low_cpu_mem_usage = True
)

# Correction Task: "text-generation"
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map = "auto"
)


# ──────────────────────────────────────────────
# 🧠 LOGIQUE RAG (VOTRE ANCIEN CODE ADAPTÉ)
# ──────────────────────────────────────────────

def semantic_chunking(text, threshold=0.75, max_len=400):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    if len(sentences) < 2: return sentences or [text]

    embs = embed_model.encode(sentences, convert_to_tensor=True)
    sims = [util.cos_sim(embs[i], embs[i + 1]).item() for i in range(len(embs) - 1)]

    chunks, current = [], sentences[0]
    for i, sim in enumerate(sims):
        next_s = sentences[i + 1]
        if sim < threshold or len(current) + len(next_s) > max_len:
            chunks.append(current.strip())
            current = next_s
        else:
            current += " " + next_s
    if current: chunks.append(current.strip())
    return chunks


def hyde_embedding(question):
    # On utilise Qwen pour générer l'hypothèse (HyDE)
    hypo_prompt = f"Réponds brièvement en français à cette question : {question}"
    messages = [{"role": "user", "content": hypo_prompt}]

    # Utilisation du template de chat pour Qwen
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    hypo_out = generator(full_prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
    hypo = hypo_out.split("<|im_start|>assistant")[-1].strip()

    combined = question + " " + hypo
    return embed_model.encode(combined, convert_to_tensor=True)


def hybrid_retrieve(question, chunks, q_embedding, top_k=5, rerank_top=3, threshold=0.25):
    corpus_embs = embed_model.encode(chunks, convert_to_tensor=True)
    dense_scores = util.cos_sim(q_embedding, corpus_embs)[0].cpu().numpy()
    dense_ranked = np.argsort(dense_scores)[::-1]

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    def rrf_score(ranks, k=60):
        return sum(1.0 / (k + r) for r in ranks)

    rrf = {}
    for rank, idx in enumerate(dense_ranked[:top_k]): rrf[idx] = rrf.get(idx, 0) + rrf_score([rank])
    for rank, idx in enumerate(bm25_ranked[:top_k]): rrf[idx] = rrf.get(idx, 0) + rrf_score([rank])

    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    candidates = [chunks[i] for i, _ in fused[:top_k] if dense_scores[i] >= threshold]

    if not candidates: candidates = [chunks[dense_ranked[0]]]

    if len(candidates) > 1:
        pairs = [[question, c] for c in candidates]
        re_scores = reranker.predict(pairs)
        candidates = [c for _, c in sorted(zip(re_scores, candidates), reverse=True)[:rerank_top]]
    return candidates


# ──────────────────────────────────────────────
# 💬 GÉNÉRATION FINALE (ADAPTATION QWEN)
# ──────────────────────────────────────────────

def chatbot_response(question, text):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = semantic_chunking(text)
    if not chunks: return "Je n'ai pas assez de texte pour répondre."

    # Retrieval
    q_emb = hyde_embedding(question)
    context_chunks = hybrid_retrieve(question, chunks, q_emb)
    context = "\n".join(context_chunks)

    # Construction du message pour Qwen (Chat Format)
    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant précis. Utilise UNIQUEMENT le contexte fourni pour répondre en français. Si la réponse n'est pas dans le texte, dis que tu ne sais pas."
        },
        {
            "role": "user",
            "content": f"Contexte :\n{context}\n\nQuestion : {question}"
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Génération avec Qwen
    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.3,  # Bas pour rester factuel
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    raw_output = result[0]["generated_text"]
    final_answer = raw_output.split("<|im_start|>assistant")[-1].strip()

    return final_answer


if __name__ == "__main__":
    sample_text = "Le projet Assistant AI 2026 est une application de gestion de cours utilisant Python et Hugging Face."
    print(chatbot_response("C'est quoi le projet 2026 ?", sample_text))