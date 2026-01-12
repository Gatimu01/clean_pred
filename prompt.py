import requests
import json
import json5
import re
import time
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed



class TitlePredictor:
    """
    TitlePredictor predicts plausible future academic paper titles for groups of authors
    using a locally hosted Large Language Model (LLM) served via Ollama.

    The class is designed for large-scale scientific forecasting experiments where:
    - Each input consists of an author group and their past paper titles.
    - The model is prompted to generate a realistic title the group could coauthor in a future year.
    - The output is strictly parsed as JSON to ensure structured, machine-readable results.

    Core features:
    - Single author-group inference via HTTP requests to Ollama.
    - Parallel execution using thread pooling for higher throughput.
    - Batched processing with automatic checkpointing to disk for fault tolerance.
    - Robust output cleaning and JSON extraction (supports markdown and minor formatting errors).
    - Configurable generation parameters (temperature, context length, GPU usage, batch size, etc.).

    Typical usage:
        predictor = TitlePredictor(model_name="Llama3.1:8B")
        results = predictor.run_prompt_batched(authors_list, titles_list)

    Requirements:
        - Ollama server running locally at http://localhost:11434
        - Python packages: requests, json5

    Intended use:
        Research on LLM-based scientific trajectory forecasting, title prediction,
        and retrieval-style evaluation of future publication modeling.

    Author:
        Maurine Gatimu
    """

    def __init__(
        self,
        model_name: str = "Llama3.1:8B",
        num_predict: int = 1000,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        num_thread: int = 2,
        num_gpu: int = -1,
        num_batch: int = 1,
    ):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

        # model config
        self.options = {
            "num_predict": num_predict,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_thread": num_thread,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
        }

    # ----------------------------------------------------------
    # 🔹 Single Author Group Prediction
    # ----------------------------------------------------------
    def process_single_author_group(
        self,
        author_group: List[str],
        past_titles: list
    ) -> Optional[str]:

        prompt = f"""You are given an author group: {', '.join(author_group)}.
These authors have collaborated on some papers as a group before January 2024, and they have also authored papers individually or in smaller subsets. Below are the paper details associated with this group's works (team papers + individual/subset papers):


{past_titles}
Using the details listed above, and using ONLY knowledge available before January 1, 2024, predict the most plausible title of a new paper that the entire group of authors would likely coauthor in 2024.
Title must be a short, realistic academic paper title.


Do NOT include any <think> sections or reasoning text.
Return ONLY valid JSON in 1–2 lines, no explanations, no markdown, no commentary.


Strictly follow this format:
{{
 "predicted_paper": {{
   "title": "<predicted title here>"
 }}
}}

"""



        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": self.options,
                },
                timeout=1000,
            )

            data = response.json()
            output = data.get("response", "")

            # Remove <think> content
            cleaned = re.split(r"<think>.*?</think>", output, flags=re.DOTALL)[-1].strip()

            # Extract JSON
            match = re.search(r"```json\n(.*?)\n```", cleaned, flags=re.DOTALL)
            if not match:
                match = re.search(r"\{[\s\S]*\}", cleaned)

            if match:
                json_text = re.sub(r"```|json", "", match.group(0)).strip()
                prediction = json5.loads(json_text)
                return prediction.get("predicted_paper", {}).get("title")

            print(f"No JSON found for authors: {author_group}")
            return None

        except Exception as e:
            print(f"Request error for {author_group}: {e}")
            return None

    # ----------------------------------------------------------
    # 🔹 Parallel Execution
    # ----------------------------------------------------------
    def run_prompt_parallel(
        self,
        authors_list: List[List[str]],
        titles_list: List[list],
        max_workers: int = 3,
    ) -> List[Optional[str]]:

        results = [None] * len(authors_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.process_single_author_group, authors_list[i], titles_list[i]): i
                for i in range(len(authors_list))
            }

            start = time.time()
            completed = 0

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    result = future.result()
                    results[idx] = result
                    completed += 1

                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = len(authors_list) - completed
                    eta = remaining / rate if rate > 0 else 0

                    # print(
                    #     f"✓ {completed}/{len(authors_list)} | "
                    #     f"Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f} min | "
                    #     f"Elapsed: {elapsed/60:.1f} min"
                    # )

                except Exception as e:
                    print(f"Error at index {idx}: {e}")

        return results

    # ----------------------------------------------------------
    # 🔹 Batched Execution + Saving Results
    # ----------------------------------------------------------

    def run_prompt_batched(
        self,
        authors_list: List[List[str]],
        titles_list: List[list],
        batch_size: int = 30,
        max_workers: int = 3,
        output_dir: str = "title_predictions",
        year_label: str = "run"
    ) -> List[Optional[str]]:

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{year_label}_all_results.json")

        # ---------------------------------------------------
        # 1. Load existing results to resume if applicable
        # ---------------------------------------------------
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"🔄 Resuming — {len(all_results)} results already exist.")
        else:
            all_results = []
            print("🆕 Starting fresh — no previous results found.")

        start_index = len(all_results)

        if start_index >= len(authors_list):
            print("✅ All authors already processed. Nothing to run.")
            return all_results

        # ---------------------------------------------------
        # 2. Loop through remaining items in batches
        # ---------------------------------------------------
        total_batches = (len(authors_list) - start_index + batch_size - 1) // batch_size
        batch_id = 1

        for i in range(start_index, len(authors_list), batch_size):
            end = min(i + batch_size, len(authors_list))

            batch_authors = authors_list[i:end]
            batch_titles = titles_list[i:end]

            print(f"\n📦 Processing batch {batch_id}/{total_batches} ({i} → {end-1})")

            # Run prediction for this batch
            batch_results = self.run_prompt_parallel(
                batch_authors, batch_titles, max_workers=max_workers
            )

            # ---------------------------------------------------
            # 3. Append batch results into the unified result list
            # ---------------------------------------------------
            all_results.extend(batch_results)

            # ---------------------------------------------------
            # 4. Save-progress after each batch
            # ---------------------------------------------------
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

            print(f"💾 Updated results saved to {output_file} "
                f"({len(all_results)} total so far)")

            batch_id += 1

        print(f"\n✅ Finished! All results saved to {output_file}")
        return all_results




# ----------------------------------------------------------
# Example Usage
# ----------------------------------------------------------
if __name__ == "__main__":
    predictor = TitlePredictor(model_name="Llama3.1:8B")

    # Dummy example
    authors = [["Alice", "Bob"], ["Carol", "Dave"]]
    titles = [["Deep Neural Networks", "Graph Models"], ["Quantum Learning"]]

    res = predictor.run_prompt_batched(
        authors_list=authors,
        titles_list=titles,
        batch_size=1,
        year_label="debug_run"
    )

    print(res)
