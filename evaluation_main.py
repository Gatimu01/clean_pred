# main.py
import argparse
import numpy as np

from evaluation import TitleForecastEvaluator, load_jsonl, load_json, flatten_2024_corpus, build_corpus_index, clean_titles_flat, filter_pairs, summarize_ranks


def main(args):
    print("Loading data...")
    records = load_jsonl(args.data, limit=args.limit)

    papers_2024 = flatten_2024_corpus(records)
    idx = build_corpus_index(papers_2024)

    true_titles = clean_titles_flat(load_json(args.true_titles))
    pred_titles = clean_titles_flat(load_json(args.predictions))

    true_clean, pred_clean = filter_pairs(true_titles, pred_titles)

    if args.max_eval:
        true_clean = true_clean[:args.max_eval]
        pred_clean = pred_clean[:args.max_eval]

    print(f"Evaluating {len(true_clean)} examples")

    evaluator = TitleForecastEvaluator()

    print("Building candidate embeddings...")
    evaluator.build_candidate_embeddings(idx.candidate_titles, batch_size=args.batch_size)

    if args.use_keybert:
        evaluator.enable_keybert()

    print("Running ranking evaluation...")
    ranks = evaluator.rank_many(
        pred_titles=pred_clean,
        true_titles=true_clean,
        idx=idx,
        papers_2024=papers_2024,
        use_oracle_categories=args.oracle_categories,
        keybert_shrink=args.use_keybert,
        bm25_rerank=args.use_bm25
    )

    print("\nRanking summary:")
    print(summarize_ranks(ranks))

    if args.compute_similarity:
        print("\nComputing SPECTER similarity...")
        sims = evaluator.specter_pairwise_scores(pred_clean, true_clean)
        print("Average cosine similarity:", float(np.mean(sims)))

    if args.compute_bertscore:
        print("\nComputing BERTScore...")
        P, R, F1 = evaluator.bertscore(pred_clean, true_clean)
        print({"P": P, "R": R, "F1": F1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Title Forecast Evaluation")

    parser.add_argument("--data", required=True, help="JSONL file with author groups")
    parser.add_argument("--predictions", required=True, help="Predicted titles JSON")
    parser.add_argument("--true_titles", required=True, help="True titles JSON")

    parser.add_argument("--limit", type=int, default=10000, help="Max records to load from JSONL")
    parser.add_argument("--max_eval", type=int, default=1000, help="Max pairs to evaluate")

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--oracle_categories", action="store_true", help="Use true categories for pool (upper bound)")
    parser.add_argument("--use_keybert", action="store_true")
    parser.add_argument("--use_bm25", action="store_true")

    parser.add_argument("--compute_similarity", action="store_true")
    parser.add_argument("--compute_bertscore", action="store_true")

    args = parser.parse_args()
    main(args)
