

import os
from datasets import load_dataset, concatenate_datasets


def main():

    out_dir = "../../../data/train_data"
    os.makedirs(out_dir, exist_ok=True)

    out_parquet = os.path.join(out_dir, "combined_human.parquet")
    out_csv = os.path.join(out_dir, "combined_human.csv")


    ds_news = load_dataset(
        "IlyaGusev/ru_news",
        split="train",
        revision="refs/convert/parquet",
    )

    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    ds_dialogues = load_dataset("Den4ikAI/russian_dialogues", split="train")
    ds_sentiment = load_dataset("MonoHime/ru_sentiment_dataset", split="train")


    def extract_text_news(example):
        return {
            "text": example.get("text", ""),
            "label": "human",
            "initial_dataset": "ru_news",
        }

    def extract_text_wiki(example):
        return {
            "text": example.get("text", ""),
            "label": "human",
            "initial_dataset": "wikipedia_ru_20231101",
        }

    def extract_text_dialogues(example):
        text = example.get("dialogue", example.get("text", ""))
        return {
            "text": text,
            "label": "human",
            "initial_dataset": "russian_dialogues",
        }

    def extract_text_sentiment(example):
        text = example.get("text", example.get("sentence", ""))
        return {
            "text": text,
            "label": "human",
            "initial_dataset": "ru_sentiment_dataset",
        }


    ds_news_clean = ds_news.map(extract_text_news, remove_columns=ds_news.column_names)
    ds_wiki_clean = ds_wiki.map(extract_text_wiki, remove_columns=ds_wiki.column_names)
    ds_dialogues_clean = ds_dialogues.map(extract_text_dialogues, remove_columns=ds_dialogues.column_names)
    ds_sentiment_clean = ds_sentiment.map(extract_text_sentiment, remove_columns=ds_sentiment.column_names)


    combined_human = concatenate_datasets([
        ds_news_clean,
        ds_wiki_clean,
        ds_dialogues_clean,
        ds_sentiment_clean,
    ])


    print("Размер:", combined_human.num_rows)
    print("Пример:", combined_human[0])
    print("Features:", combined_human.features)


    combined_human.to_parquet(out_parquet)
    print(f"✅ Saved Parquet: {out_parquet}")


if __name__ == "__main__":
    main()
