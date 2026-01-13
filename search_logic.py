import pandas as pd
import numpy as np
import re
import warnings
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, CrossEncoder, util

warnings.filterwarnings('ignore')

# --- DATA PREPARATION ---
# Make sure service_data.csv is in the same folder!
df_raw = pd.read_csv('service_data.csv')

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    return text.strip()

df_raw['CleanSnippet'] = (df_raw['ShortDescription'].fillna('') + " " + df_raw['ContentText'].fillna('')).apply(clean_text)
snippet_map = df_raw[df_raw['CleanSnippet'].str.len() > 30][['ServiceName', 'ServiceSKU', 'CleanSnippet']].drop_duplicates()

# --- MODELS ---
bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
snippet_embeddings = bi_encoder.encode(snippet_map['CleanSnippet'].tolist())

# --- SEARCH FUNCTION ---
def search_service_optimized(query, top_n=5):
    query = query.strip()
    names = snippet_map['ServiceName'].unique().tolist()
    fuzzy_matches = process.extract(query, names, scorer=fuzz.WRatio, limit=3, score_cutoff=85)

    if fuzzy_matches:
        results = []
        for name, score, _ in fuzzy_matches:
            row = snippet_map[snippet_map['ServiceName'] == name].iloc[0]
            results.append({'ServiceName': name, 'ServiceSKU': row['ServiceSKU'], 'Confidence': f"{score}%"})
        return results

    query_emb = bi_encoder.encode(query)
    hits = util.semantic_search(query_emb, snippet_embeddings, top_k=10)[0]
    
    results_list = []
    for hit in hits:
        idx = hit['corpus_id']
        results_list.append({
            'ServiceName': snippet_map.iloc[idx]['ServiceName'],
            'ServiceSKU': snippet_map.iloc[idx]['ServiceSKU'],
            'Snippet': snippet_map.iloc[idx]['CleanSnippet']
        })
    return results_list[:top_n]

# --- RECOMMENDATION FUNCTION ---
def get_recommendations(service_sku, top_n=3):
    idx_list = snippet_map.index[snippet_map['ServiceSKU'] == service_sku].tolist()
    if not idx_list: return []
    
    target_idx = idx_list[0]
    target_emb = snippet_embeddings[target_idx]
    cos_scores = util.cos_sim(target_emb, snippet_embeddings)[0]
    
    top_results = np.argpartition(-cos_scores, range(top_n + 1))[1:top_n + 1]
    
    recs = []
    for idx in top_results:
        recs.append({
            "ServiceName": snippet_map.iloc[int(idx)]['ServiceName'],
            "ServiceSKU": snippet_map.iloc[int(idx)]['ServiceSKU']
        })
    return recs