"""Flatten the Techmap US job postings JSON dump into a CSV.

The Kaggle file (techmap-jobs_us_2023-05-05.json, 4.4 GB) has one nested JSON
document per line, and each job board stores fields in a different place. This
script streams it straight out of the zip (never loading it all into memory),
pulls the same set of fields from every posting with fallbacks, and writes:

* techmap_flat.csv      one row per posting (metadata columns)
* techmap_text.jsonl    id + full description text (kept separate so the CSV stays small)

Example:
    python src/data/flatten_techmap.py --zip data/raw/us_jobpostings_kaggle.zip --out data/raw
"""

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


def get(d, *path):
    """Safely walk nested dicts; returns None if any level is missing."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def date(value):
    """Unwrap MongoDB-style dates ({'$date': '...'})."""
    return value.get("$date") if isinstance(value, dict) else value


COLUMNS = ["id", "source", "idInSource", "title", "workType", "employmentType", "company",
           "city", "state", "country", "url", "dateCreated", "datePosted", "dateScraped",
           "validThrough", "text_len", "text_hash", "category", "salary_text", "careerLevel",
           "industry"]


def flatten(doc):
    """Turn one Techmap JSON document into a flat row (list in COLUMNS order)."""
    pos = doc.get("position") or {}
    schema = get(doc, "json", "schemaOrg") or {}
    company = doc.get("orgCompany") or {}
    address = doc.get("orgAddress") or {}
    salary = doc.get("salary") if isinstance(doc.get("salary"), dict) else {}
    tags = get(doc, "orgTags", "CATEGORIES") or []
    text = doc.get("text") or ""
    return [
        get(doc, "_id", "$oid"), doc.get("source"), doc.get("idInSource"),
        pos.get("name") or doc.get("name"), pos.get("workType"), schema.get("employmentType"),
        company.get("nameOrg") or company.get("name") or get(schema, "hiringOrganization", "name"),
        address.get("city"), address.get("state"), address.get("countryCode"), doc.get("url"),
        date(doc.get("dateCreated")), schema.get("datePosted"), date(doc.get("dateScraped")),
        schema.get("validThrough"), len(text),
        hashlib.md5(text.strip().lower().encode()).hexdigest() if text else None,
        tags[0] if tags else None, salary.get("text"), pos.get("careerLevel"),
        json.dumps(schema.get("industry"))[:150] if schema.get("industry") else None,
    ], text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", required=True, type=Path, help="Kaggle zip containing the JSON file")
    ap.add_argument("--out", default="data/raw", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = 0
    with zipfile.ZipFile(args.zip) as z, z.open(z.namelist()[0]) as fh, \
            open(args.out / "techmap_flat.csv", "w", newline="") as f_csv, \
            open(args.out / "techmap_text.jsonl", "w") as f_txt:
        writer = csv.writer(f_csv)
        writer.writerow(COLUMNS)
        for line in fh:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            row, text = flatten(doc)
            writer.writerow(row)
            f_txt.write(json.dumps({"id": row[0], "text": text}) + "\n")
            n += 1
    print(f"wrote {n:,} postings to {args.out}")


if __name__ == "__main__":
    main()
