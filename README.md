**LLM-based Scientific Title Forecasting Pipeline**

This repository contains a complete pipeline for forecasting future scientific paper titles using large language models (LLMs), based on historical author publication records and automatically extracted future work sections from arXiv papers.

The system supports:

  Large-scale arXiv corpus collection (API-safe, resumable)
  
  Structured dataset construction (author groups + past papers)
  
  Automatic extraction of future work sections from LaTeX / PDF (GROBID fallback)
  
  Dataset enrichment with extracted future work
  
  Parallel LLM inference using Ollama
  
  Resume-safe batching and evaluation utilities


.

├── arxiv_corpus.py           # arXiv API downloader (monthly corpus builder)

├── paper_load_fw.py          # Loader with future-work support

├── future_work_extractor.py # LaTeX + GROBID future work extraction pipeline

├── prompt.py                 # TitlePredictor (LLM interface)

├── Evaluation/

    ├── evaluator.py              # Evaluation logic

    ├── metrics.py                # Similarity metrics (BERTScore, cosine, BM25)

    └── corpus.py                 # Corpus utilities

    ├── evaluator.py              # Evaluation logic

    ├── metrics.py                # Similarity metrics (BERTScore, cosine, BM25)

    └── corpus.py                 # Corpus utilities

├── main.py                   # End-to-end forecasting pipeline

├── data/

    ├── raw/                  # Raw arXiv JSONL files

    ├── processed/            # Structured datasets

    └── predictions/          # LLM outputs

└── README.md



**1. Build arXiv Corpus**

Uses the official arXiv Atom API with:
  Month splitting to bypass 2000-result limit
  
  Pagination
  
  Deduplication by arXiv ID
  
  For example:

    from arxiv_corpus import ArxivCorpusFetcher
    
    fetcher = ArxivCorpusFetcher()
    fetcher.fetch_month(2023, 1, save_file="data/raw/2023_01.jsonl", return_list=False)

