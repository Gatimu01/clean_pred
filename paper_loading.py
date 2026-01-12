import json
import os
from datetime import datetime
from typing import List, Dict, Any


class PaperLoader:
    """
    Improved loader:
    - Extracts year and YYYY-MM date
    - Preserves full past paper details
    - Prepares merged structure with dates
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw = self._load_jsonl(file_path)

        (
            self.authors_list,
            self.current_titles,
            self.past_titles_raw,
            self.paper_details,
            self.dates,
            self.categories
        ) = self._extract_components()

        self.merged_records = self._merge_all_info()

    # ------------------------------
    # JSONL Loader
    # ------------------------------
    @staticmethod
    def _load_jsonl(path: str, limit: int = 15000) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")

        papers = []
        count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= limit:
                    break  # stop reading after limit lines

                try:
                    papers.append(json.loads(line.strip()))
                    count += 1
                except json.JSONDecodeError as e:
                    print("Skipping invalid JSON:", e)

        return papers


    # ------------------------------
    # Extract components
    # ------------------------------
    def _extract_components(self):
        authors_list = [p.get("author_group", []) for p in self.raw]



        current_titles = [
            [c.get("title", "") for c in p.get("current_papers", []) if c.get("title")]
            for p in self.raw
        ]

        # Extract title-only version
        past_titles_raw = [
            {f"Title {i+1}": p.get("title", "")
             for i, p in enumerate(entry.get("past_papers", []))}
            for entry in self.raw
        ]

        # Extract full details (title, abstract, date, year)
        paper_details = []
        dates_list = []

        for entry in self.raw:
            details = {}
            dates = {}

            for i, paper in enumerate(entry.get("past_papers", [])):
                idx = i + 1

                pub = paper.get("published")
                year = None
                date_str = None

                if pub:
                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    year = dt.year
                    date_str = f"{dt.year}-{dt.month:02d}"

                details[f"Title {idx}"] = paper.get("title", "")
                details[f"Abstract {idx}"] = paper.get("abstract", "")
                details[f"Date {idx}"] = date_str
                details[f"Year {idx}"] = year
                details[f"Arxiv_id {idx}"] = paper.get("arxiv_id", "")
                details[f"FutureWork {idx}"] = paper.get("future_work_text", "")
                

                dates[f"Date {idx}"] = date_str

            paper_details.append(details)
            dates_list.append(dates)

        categories = [p.get("main_category", "") for p in self.raw]

        return authors_list, current_titles, past_titles_raw, paper_details, dates_list, categories

    # ------------------------------
    # Merge all information
    # ------------------------------
    def _merge_all_info(self):
        merged_records = []

        for i in range(len(self.authors_list)):
            merged_papers = {}

            # reconstruct merged past papers with full details
            for key in self.past_titles_raw[i]:
                idx = key.split()[-1]

                merged_papers[idx] = {
                    "title": self.paper_details[i].get(f"Title {idx}"),
                    "abstract": self.paper_details[i].get(f"Abstract {idx}"),
                    "date": self.paper_details[i].get(f"Date {idx}"),
                    "year": self.paper_details[i].get(f"Year {idx}"),
                    "arxiv_id": self.paper_details[i].get(f"Arxiv_id {idx}"),
                    "future_work_text": self.paper_details[i].get(f"FutureWork {idx}")
                }

            merged_records.append({
                "authors": self.authors_list[i],
                "past_papers": merged_papers,
                "current_titles": self.current_titles[i],
                "category": self.categories[i]
            })

        return merged_records

    # ------------------------------
    # Final usable outputs
    # ------------------------------

    def get_full_record(self):
        return self.merged_records
    
    def get_authors(self):
        return [r["authors"] for r in self.merged_records]

    def get_past_titles(self):
        """
        Simpler version of past paper titles.
        """
        cleaned = []
        for record in self.merged_records:
            cleaned.append({
                f"Title {i}": p["title"]
                for i, p in record["past_papers"].items()
            })
        return cleaned
    
    

    def get_sorted_history(self, min_year: int = None):
        """
        Return sorted past paper history in the format:
        'YYYY-MM: Title', sorted newest → oldest.

        If min_year is provided, only include papers from that year onward.
        """
        all_sorted = []

        for record in self.merged_records:
            past = record["past_papers"]

            # Convert into sortable list
            entries = []
            for idx, p in past.items():
                date = p.get("date")    # YYYY-MM
                title = p.get("title")
                year = p.get("year")

                # Skip if missing date
                if not date or not year is None:
                    continue

                # Optional filter
                if min_year is not None and year < min_year:
                    continue

                entries.append((date, title))

            # Sort by date desc (YYYY-MM string formats sort correctly lexicographically)
            entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)

            # Convert to output format
            formatted = [f"{date}: {title}" for date, title in entries_sorted]

            all_sorted.append(formatted)

        return all_sorted
    
    def get_sorted_history_titles(self, min_year: int = None):
        """
        Return sorted past paper history in the format:
        'YYYY-MM: Title', sorted newest → oldest.

        If min_year is provided, only include papers from that year onward.
        """
        all_sorted = []

        for record in self.merged_records:
            past = record["past_papers"]

            # Convert into sortable list
            entries = []
            for idx, p in past.items():
                date = p.get("date")    # YYYY-MM
                title = p.get("title")
                year = p.get("year")

                # Skip if missing date
                if not date or not year:
                    continue

                # Optional filter
                if min_year is not None and year < min_year:
                    continue

                entries.append((date, title))

            # Sort by date desc (YYYY-MM string formats sort correctly lexicographically)
            entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)

            # Convert to output format
            formatted = [f"{title}" for date, title in entries_sorted]

            all_sorted.append(formatted)

        return all_sorted


    def get_history_with_future_work(self):
        """
        For each record:
        - Find max year among past papers
        - For papers in max year:
            return Title, Year, FutureWork (fallback to Abstract)
        - For older papers:
            return Title, Year only
        """

        all_records = []

        for record in self.merged_records:
            past = record["past_papers"]

            # ---- find max year ----
            years = [p.get("year") for p in past.values() if p.get("year") is not None]
            max_year = max(years) if years else None

            processed = []

            for idx, p in past.items():
                title = p.get("title")
                year = p.get("year")

                if year is None or title is None:
                    continue

                # latest year → include future work 
                if year == max_year:
                    future_text = p.get("future_work_text")

                    if not future_text:
                        future_text = p.get("abstract")  # fallback

                    processed.append({
                        "title": title,
                        "year": year,
                        "future_work": future_text
                    })

                # ---- older papers ----
                else:
                    processed.append({
                        "title": title,
                        "year": year
                    })

            all_records.append({
                # "authors": record["authors"],
                # "category": record["category"],
                "papers": processed
            })

        return all_records


    def get_sorted_history_titlesAbstract(self, min_year: int = None):
        """
        Return sorted past paper history in the format:
        'YYYY-MM: Title', sorted newest → oldest.

        If min_year is provided, only include papers from that year onward.
        """
        all_sorted = []

        for record in self.merged_records:
            past = record["past_papers"]

            # Convert into sortable list
            entries = []
            for idx, p in past.items():
                date = p.get("date")    # YYYY-MM
                title = p.get("title")
                year = p.get("year")
                abstract=p.get("abstract")

                # Skip if missing date
                if not date or not year:
                    continue

                # Optional filter
                if min_year is not None and year < min_year:
                    continue

                entries.append(( date, title, abstract))

            # Sort by date desc (YYYY-MM string formats sort correctly lexicographically)
            entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)

            # Convert to output format
            formatted = [f"{title}. {abstract}" for date, title, abstract in entries_sorted]

            all_sorted.append(formatted)

        return all_sorted

