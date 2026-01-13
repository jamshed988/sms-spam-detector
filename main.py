from fastapi import FastAPI
from search_logic import search_service_optimized, get_recommendations

app = FastAPI(title="Expert Service Search & Recs API")

@app.get("/")
def health_check():
    return {"status": "Online", "message": "Search Engine Ready"}

@app.get("/search")
def search(query: str, limit: int = 5):
    """Search for services using semantic AI."""
    results = search_service_optimized(query, top_n=limit)
    return {"query": query, "results": results}

@app.get("/recommend/{sku}")
def recommend(sku: str, limit: int = 3):
    """Get recommendations similar to a specific Service SKU."""
    recommendations = get_recommendations(sku, top_n=limit)
    return {"original_sku": sku, "recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)