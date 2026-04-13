import os
import json
import base64
from urllib import response
import faiss
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---- CONFIG ----
GROQ_API_KEY  = "your_groq_api_key_here"
MEMES_FOLDER  = "./memes"
METADATA_FILE = "./meme_metadata.json"
TOP_K         = 5
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

# ---- MODELS ----
print("Loading models...")
# embedder = SentenceTransformer('all-mpnet-base-v2')
embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
client   = Groq(api_key=GROQ_API_KEY)
print("Models ready!\n")


# 1. DUPLICATE FILE DETECTION

def get_file_hash(filepath):
    """
    Reads every byte of the image and produces a unique MD5 string.
    Same image bytes = same hash, even if filename is different.
    Used to skip duplicate image files before indexing.
    """
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# 2. VISION — describe one meme

def describe_meme(image_path):
    """
    Converts image to base64, sends to Groq vision model.
    Returns structured dict: title, category, keywords, funniness, description.
    Result is saved to JSON so we never call the API twice for the same meme.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext        = image_path.lower().split('.')[-1]
    media_type = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"

    response = client.chat.completions.create(
        model    = "meta-llama/llama-4-scout-17b-16e-instruct",
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": """You are a meme expert. Analyze this meme very carefully.
Respond ONLY with this exact JSON, no extra text:

{
  "title": "creative specific title that captures the joke",
  "category": "pick EXACTLY one: dark, dad_joke, wholesome, relatable, political, sports, gaming, programming, animals, other",
  "keywords": ["specific_keyword1", "specific_keyword2", "specific_keyword3", "specific_keyword4", "specific_keyword5"],
  "funniness": 6,
  "description": "Write 4 sentences: (1) exactly what the image shows visually. (2) exactly what text appears on the meme word by word. (3) what type of humor this is and who would laugh at it. (4) what specific situations, feelings, or topics this meme relates to."
}

Rules you MUST follow:
- category MUST reflect the PRIMARY humor type, not a secondary one
- keywords MUST be specific search terms someone would type to find THIS meme
- funniness MUST be honest: 1-3 = barely funny, 4-6 = average, 7-8 = genuinely funny, 9-10 = extremely funny. Do NOT default to 7.
- description MUST mention the exact words/text visible on the meme
- Respond with ONLY the JSON. No markdown. No explanation."""
                }
            ]
        }],
        max_tokens = 500
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# 3. LOAD / CREATE METADATA

def load_or_create_metadata():
    """
    Loads existing descriptions from meme_metadata.json.
    For any new image files not yet described — describes them.
    Skips duplicate image files using MD5 hash comparison.
    Saves all metadata back to JSON after processing.
    """
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} existing meme descriptions.")
    else:
        metadata = {}

    seen_hashes = {v["hash"] for v in metadata.values() if "hash" in v}
    duplicates  = 0
    new_count   = 0

    all_files = [
        f for f in os.listdir(MEMES_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ]
    print(f"Found {len(all_files)} image files in folder.")

    for filename in all_files:
        # already described — skip
        if filename in metadata:
            continue

        filepath  = os.path.join(MEMES_FOLDER, filename)
        file_hash = get_file_hash(filepath)

        # same image bytes as an already-processed file — skip
        if file_hash in seen_hashes:
            print(f"  Duplicate file skipped: {filename}")
            duplicates += 1
            continue

        print(f"  Describing: {filename}...")
        try:
            info = describe_meme(filepath)
            metadata[filename] = {
                "filename":    filename,
                "path":        os.path.abspath(filepath),
                "hash":        file_hash,
                "title":       info.get("title",       filename),
                "category":    info.get("category",    "other"),
                "keywords":    info.get("keywords",    []),
                "funniness":   info.get("funniness",   5),
                "description": info.get("description", "")
            }
            seen_hashes.add(file_hash)
            new_count += 1
            print(f"    Title:    {metadata[filename]['title']}")
            print(f"    Category: {metadata[filename]['category']}")
            print(f"    Funny:    {metadata[filename]['funniness']}/10")
        except Exception as e:
            print(f"    Failed to describe {filename}: {e}")

    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone — {new_count} new, {duplicates} file duplicates skipped.")
    print(f"Total in system: {len(metadata)}\n")
    return metadata


# 4. BUILD FAISS INDEX

def build_index(metadata):
    """
    For each meme builds a rich search string.
    Category repeated 3x and keywords repeated 2x so they
    dominate the embedding — ensuring category-based searches
    find the right meme type first.
    Stores normalized vectors in FAISS IndexFlatIP
    which gives cosine similarity scores.
    """
    meme_list    = list(metadata.values())
    search_texts = []

    for m in meme_list:
        keywords_str = ', '.join(m['keywords'])
        rich_text = (
            f"Category: {m['category']}. {m['category']}. {m['category']}.\n"
            f"Keywords: {keywords_str}. {keywords_str}.\n"
            f"Title: {m['title']}. {m['title']}.\n"
            f"Description: {m['description']}\n"
            f"Funniness: {m['funniness']} out of 10"
        )
        search_texts.append(rich_text)

    print(f"Embedding {len(search_texts)} memes...")
    vectors = embedder.encode(
        search_texts,
        show_progress_bar    = True,
        normalize_embeddings = True
    ).astype('float32')

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"Index ready — {index.ntotal} memes indexed.\n")
    return index, meme_list


# 5. VECTOR SEARCH — with result deduplication

def search_memes(query, index, meme_list, k=TOP_K):
    """
    Embeds the query and finds closest memes in FAISS.
    Fetches double the needed amount then deduplicates by filename
    so the same meme never appears twice in results.
    """
    q_vec = embedder.encode(
        [query],
        normalize_embeddings = True
    ).astype('float32')

    scores, indices = index.search(q_vec, k * 2)  # fetch extra to cover dups

    results = []
    seen    = set()  # filenames already added to results

    for i, idx in enumerate(indices[0]):
        m        = meme_list[idx]
        filename = m["filename"]

        if filename in seen:
            continue  # skip duplicate result

        seen.add(filename)
        results.append({
            "rank":        len(results) + 1,
            "filename":    filename,
            "path":        m["path"],
            "title":       m["title"],
            "category":    m["category"],
            "keywords":    m["keywords"],
            "funniness":   m["funniness"],
            "description": m["description"],
            "score":       round(float(scores[0][i]) * 100, 1)
        })

        if len(results) == k:
            break

    return results


# 6. LLM RERANKING — accuracy booster

def rerank_with_llm(query, results):
    """
    Vector search finds candidates by math.
    LLM reads all candidates and reorders by strict relevance rules:
    1. Category match first
    2. Keywords match second
    3. Description relevance third
    4. Funniness ignored for ranking
    Falls back to original order if LLM call fails.
    """
    candidates = ""
    for r in results:
        candidates += (
            f"\nMeme #{r['rank']} — {r['title']}\n"
            f"  Category: {r['category']}\n"
            f"  Keywords: {', '.join(r['keywords'])}\n"
            f"  Description: {r['description']}\n"
            f"---"
        )

    prompt = f"""You are a meme search engine. A user searched for: "{query}"

Here are the candidate memes:
{candidates}

Rank them from BEST to WORST match for the user's search.

Strict priority rules:
1. CATEGORY match is most important
2. KEYWORDS match is second
3. DESCRIPTION relevance is third
4. IGNORE funniness — relevance only

Reply ONLY with a JSON array of meme numbers ranked best to worst.
Example: [3,1,2,4,5]
No explanation. No markdown. Just the array."""

    try:
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1
        )
        raw   = response.choices[0].message.content.strip()
        raw   = raw.replace("```json","").replace("```","").strip()
        order = json.loads(raw)

        reranked = []
        for new_rank, original_rank in enumerate(order):
            for r in results:
                if r["rank"] == original_rank:
                    r["rank"] = new_rank + 1
                    reranked.append(r)
                    break
        return reranked

    except Exception as e:
        print(f"Reranking failed, using vector order: {e}")
        return results


# 7. QUERY EXPANSION

def expand_query(query):
    """
    User types a short query like "dad jokes".
    LLM expands it to richer search terms like
    "dad_joke pun wordplay wholesome family humor groan funny"
    giving the embedder much more signal to work with.
    """
    response = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [{
            "role": "user",
            "content": f"""A user is searching for a meme: "{query}"

Expand into richer search terms including:
- the meme category (dark, dad_joke, wholesome, relatable, political, sports, gaming, programming, animals, other)
- related keywords someone would use to describe this meme type
- the mood and humor style

Reply with ONLY a single line of search terms. No explanation.
Example input: "dad jokes"
Example output: "dad_joke pun wordplay wholesome family humor groan funny father joke"

Expand: "{query}" """
        }],
        temperature = 0.1
    )
    expanded = response.choices[0].message.content.strip()
    expanded = f"Represent this sentence for searching relevant passages: {expanded}"
    print(f"  Expanded query ready.")
    return expanded


# 8. MAIN PIPELINE

def find_memes(query, index, meme_list, top_n=3):
    """
    Full pipeline called by both CLI and app.py:
    1. Expand query for richer search signal
    2. Vector search with deduplication
    3. LLM rerank by strict relevance rules
    Returns top_n unique relevant memes.
    """
    expanded = expand_query(query)
    results  = search_memes(expanded, index, meme_list)
    reranked = rerank_with_llm(query, results)
    return reranked[:top_n]


# CLI MODE

if __name__ == "__main__":
    metadata         = load_or_create_metadata()
    index, meme_list = build_index(metadata)

    print("=== CLI TEST ===")
    for q in ["dark humor", "dad joke", "relatable work meme"]:
        print(f"\nQuery: '{q}'")
        for r in find_memes(q, index, meme_list):
            print(f"  #{r['rank']} {r['title']} ({r['category']}) — {r['score']}%")

    print("\n=== INTERACTIVE ===")
    while True:
        q = input("\nSearch: ").strip()
        if q.lower() in ['quit', 'exit', 'q']:
            break
        if q:
            for r in find_memes(q, index, meme_list):
                print(f"  #{r['rank']} {r['title']}")
                print(f"      Path:  {r['path']}")
                print(f"      Match: {r['score']}%")