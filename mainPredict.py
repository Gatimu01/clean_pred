
import time
from paper_loading import PaperLoader
from prompt import TitlePredictor



def format_time(seconds):
    """Return human readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"


def main():

    overall_start = time.time()
    print("⏳ Pipeline started...")

    
    # 1. Load papers

    t0 = time.time()
    print("Loading dataset...")
    loader = PaperLoader("Data_to_run50000.jsonl")
    print(f"Load time: {format_time(time.time() - t0)}")
    
    authors = loader.get_authors()
    past_titles = loader.get_sorted_history_titlesAbstract()


    # 2. Initialize predictor
   
    t1 = time.time()
    predictor = TitlePredictor(model_name="Llama3.1:8B")
    print(f"Predictor initialized in {format_time(time.time() - t1)}")


    # 3. Run predictions
 
    t2 = time.time()
    print("🚀 Starting prediction pipeline...")


    results = predictor.run_prompt_batched(
        authors_list=authors,
        titles_list=past_titles,
        batch_size=100,
        max_workers=4,
        output_dir="predictions",
        year_label="2024_run"
    )

    print(f"Prediction stage finished in {format_time(time.time() - t2)}")

    # 4. Final summary
  
    total_time = time.time() - overall_start

    print("\n🎉 Prediction completed!")
    print(f"Saved {len(results)} predictions.")
    print(f"⏱ Total runtime: {format_time(total_time)}")


if __name__ == "__main__":
    main()
