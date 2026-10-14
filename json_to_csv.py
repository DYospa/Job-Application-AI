
import os
import pandas as pd

DATA_DIR = "/Users/doronyospa/Desktop/JobAI/data"

def json_to_csv(json_file: str, csv_file: str) -> None:
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File not found: {json_file}")

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # Try newline-delimited JSON and process it in chunks.
    try:
        first_chunk = True

        for chunk in pd.read_json(
            json_file,
            lines=True,
            chunksize=5000
        ):
            chunk.to_csv(
                csv_file,
                mode="w" if first_chunk else "a",
                index=False,
                header=first_chunk
            )
            first_chunk = False

        if first_chunk:
            print("The JSON file contains no records.")
        else:
            print(f"Conversion complete: {csv_file}")

    except ValueError as e:
        # Standard JSON needs a different approach.
        if os.path.exists(csv_file):
            os.remove(csv_file)

        raise ValueError(
            "The file may be standard JSON rather than JSONL. "
            "Check its structure before converting. "
            f"Details: {e}"
        ) from e


if __name__ == "__main__":
    json_to_csv(
        os.path.join(DATA_DIR, "techmap-jobs_us_2023-05-05.json"),
        os.path.join(DATA_DIR, "techmap-jobs_us_2023-05-05.csv")
    )
