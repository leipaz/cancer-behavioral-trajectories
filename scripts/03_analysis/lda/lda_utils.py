"""Shared helpers for LDA training and post-processing."""

from __future__ import annotations

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim import corpora
from gensim.models import LdaModel


def parse_embedding_ids(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def build_documents(df_profiles: pd.DataFrame) -> list[list[str]]:
    ids = df_profiles["embedding_ids"].apply(parse_embedding_ids)
    return [[str(token) for token in doc] for doc in ids]


def load_lda_artifacts(
    dictionary_path,
    model_path,
    profiles_csv,
) -> tuple[corpora.Dictionary, LdaModel, pd.DataFrame, list, list]:
    dictionary = corpora.Dictionary.load(str(dictionary_path))
    lda_model = LdaModel.load(str(model_path))
    df_profiles = pd.read_csv(profiles_csv)
    df_profiles["embedding_ids"] = df_profiles["embedding_ids"].apply(parse_embedding_ids)
    documents = build_documents(df_profiles)
    corpus_bow = [dictionary.doc2bow(doc) for doc in documents]
    return dictionary, lda_model, df_profiles, documents, corpus_bow


def get_top_terms_positive_probability(
    lda_model: LdaModel,
    corpus,
    dictionary: corpora.Dictionary,
    top_n: int = 10,
) -> pd.DataFrame:
    topic_term_dists = lda_model.get_topics()
    num_topics, vocab_size = topic_term_dists.shape
    vocab = [dictionary[i] for i in range(vocab_size)]
    rows = []

    for topic_idx in range(num_topics):
        topic_probs = topic_term_dists[topic_idx]
        top_indices = topic_probs.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_indices):
            rows.append(
                {
                    "topic": topic_idx,
                    "rank": rank + 1,
                    "profile": vocab[idx],
                    "probability": topic_probs[idx],
                }
            )
    return pd.DataFrame(rows)


def assign_topics_to_users(
    lda_model: LdaModel,
    corpus_bow,
    df_profiles: pd.DataFrame,
) -> pd.DataFrame:
    topic_distributions = [
        lda_model.get_document_topics(bow, minimum_probability=0.0)
        for bow in corpus_bow
    ]
    num_topics = len(lda_model.get_topics())

    topic_distributions_list = []
    for doc in topic_distributions:
        topic_probs = [0.0] * num_topics
        for topic_id, prob in doc:
            topic_probs[topic_id] = prob
        topic_distributions_list.append(topic_probs)

    predominant_topics = [max(doc, key=lambda x: x[1])[0] for doc in topic_distributions]

    result_df = pd.DataFrame(
        {
            "user_id": df_profiles["user_id"],
            "predominant_topic": predominant_topics,
            "topic_distribution": topic_distributions_list,
        }
    )
    topic_columns = [f"topic_{i}" for i in range(num_topics)]
    result_df[topic_columns] = pd.DataFrame(
        result_df["topic_distribution"].tolist(), index=result_df.index
    )
    return result_df


def plot_top_terms_by_topic_grid(
    terms_df: pd.DataFrame,
    output_path=None,
    top_n: int = 10,
) -> None:
    num_topics = terms_df["topic"].nunique()
    n_cols = 3
    n_rows = (num_topics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9))
    fig.suptitle(
        "Predominant Day Types per Pattern",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    axes = np.array(axes).flatten()
    value_col = "probability" if "probability" in terms_df.columns else "relevance"

    for topic_idx in range(num_topics):
        ax = axes[topic_idx]
        topic_data = (
            terms_df[terms_df["topic"] == topic_idx]
            .sort_values(by=value_col, ascending=False)
            .head(top_n)
        )
        sns.barplot(
            x=value_col,
            y="profile",
            data=topic_data,
            ax=ax,
            palette="mako",
            hue="profile",
            legend=False,
        )
        ax.set_title(f"Pattern {topic_idx}", fontsize=12, fontweight="semibold")
        ax.set_xlabel(value_col.capitalize())
        ax.set_ylabel("")

    for i in range(num_topics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3.0)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def harmonize_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    out = df.rename(columns={"user_id": id_col, "ID": id_col, "S_RECORD_id": id_col}, errors="ignore")
    if id_col in out.columns:
        out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    return out
