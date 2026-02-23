from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from services.nlp_processor import setup_nltk, prepare_dataframe
from services.ml_model import (
    cluster_unique_words,
    get_top_keywords_per_cluster,
    embedding_model,
)
from services.matchmaker import assign_members

app = FastAPI(
    title="Matchmaking Clustering API",
    description="API for load-balanced matchmaking.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Runs once when the server boots up."""
    print("Initializing NLTK for the API...")
    setup_nltk()


@app.post("/api/v1/cluster")
async def cluster_data(
    file: UploadFile = File(...),
    text_column: str = Form("interests"),
    num_clusters: int = Form(10),
    max_diff: int = Form(20),
):
    """Accepts a CSV, clusters the members, and returns a new CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        contents = await file.read()
        dataset = pd.read_csv(
            io.StringIO(contents.decode("utf-8")), on_bad_lines="warn"
        )

        if text_column not in dataset.columns:
            raise HTTPException(
                status_code=400, detail=f"Column '{text_column}' not found in CSV."
            )

        processed_df, unique_keywords_df = prepare_dataframe(dataset, text_column)

        kmeans, word_embeddings, clustered_words_df = cluster_unique_words(
            unique_keywords_df, num_clusters
        )

        house_labels = get_top_keywords_per_cluster(
            kmeans, word_embeddings, clustered_words_df
        )

        final_df = assign_members(
            processed_df, embedding_model, kmeans, max_diff=max_diff
        )

        final_df["cluster_label"] = final_df["assigned_cluster"].apply(
            lambda x: ", ".join(house_labels.get(x, []))
        )

        stream = io.StringIO()
        final_df.to_csv(stream, index=False)

        response_stream = io.BytesIO(stream.getvalue().encode("utf-8"))

        return StreamingResponse(
            response_stream,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=clustered_{file.filename}"
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
