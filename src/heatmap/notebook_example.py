from heatmap.analysis import run_heatmap_analysis
from heatmap.run_scientific_heatmap import load_articles
from model_service import BertAIDetector, DetectorConfig


detector = BertAIDetector(
    DetectorConfig(
        detector_type="ensemble",
        model_path="trained_models/ensemble_models/manifest.json",
        device="cpu",
        max_length=256,
        threshold=0.8,
    )
)


df_articles = load_articles(
    input_path="scientific_articles",
    text_col="text",
    article_id_col="article_id",
)


result = run_heatmap_analysis(
    detector=detector,
    df_articles=df_articles,
    out_dir="scientific_heatmap_output",
    text_col="text",
    article_id_col="article_id",
    batch_size=32,
    window_size=4,
    min_sentences=3,
    threshold=0.8,
    max_articles_in_heatmap=40,
)

print(result["global_stats"])
print(result["article_summary"].head(10))
print(result["feature_comparison"].head(10))


df_scored = result["df_scored"]

print("\n=== TOP-5 MOST AI-LIKE CHUNKS ===")
for _, row in df_scored.sort_values("p_ai", ascending=False).head(5).iterrows():
    print(f"\n[{row['chunk_id']}] p_ai={row['p_ai']:.4f}")
    print(row["chunk_text"])

print("\n=== TOP-5 LEAST AI-LIKE CHUNKS ===")
for _, row in df_scored.sort_values("p_ai", ascending=True).head(5).iterrows():
    print(f"\n[{row['chunk_id']}] p_ai={row['p_ai']:.4f}")
    print(row["chunk_text"])
