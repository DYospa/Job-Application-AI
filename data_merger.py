
import os
import pandas as pd

DATA_DIR = "/Users/doronyospa/Desktop/JobAI/data"


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"Skipping missing file: {filename}")
        return None

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"Loaded {filename}: {len(df):,} rows")
    return df


def find_column(df, candidates):
    """Find the first matching column name."""
    if df is None:
        return None

    lookup = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def aggregate_values(df, key, value, output_name):
    """Produce one row per key, combining multiple values."""
    if df is None or key not in df.columns or value not in df.columns:
        return None

    temp = df[[key, value]].dropna(subset=[key, value]).copy()
    temp[value] = temp[value].astype(str)

    return (
        temp.groupby(key)[value]
        .agg(lambda s: "; ".join(sorted(set(s))))
        .reset_index(name=output_name)
    )


def merge_lookup(base, lookup, left_key, right_key,
                 value_column, output_name):
    """Attach lookup values without multiplying job rows."""
    if lookup is None:
        return base

    if left_key not in base.columns or right_key not in lookup.columns:
        print(f"Cannot join lookup for {output_name}: key missing")
        return base

    aggregated = aggregate_values(
        lookup, right_key, value_column, output_name
    )

    if aggregated is None:
        print(f"Cannot aggregate {output_name}: column missing")
        return base

    return base.merge(
        aggregated,
        left_on=left_key,
        right_on=right_key,
        how="left",
        validate="many_to_one"
    ).drop(columns=[right_key], errors="ignore")


def main():
    # 1. Main job posting table
    jobs = load_csv("postings.csv")

    if jobs is None:
        raise FileNotFoundError(
            "postings.csv was not found in the data directory."
        )

    job_key = find_column(jobs, ["job_id", "id"])

    if job_key is None:
        raise ValueError("No job identifier found in postings.csv.")

    # 2. Company details
    companies = load_csv("companies.csv")

    if companies is not None:
        company_key = find_column(jobs, ["company_id"])
        lookup_key = find_column(companies, ["company_id", "id"])

        if company_key and lookup_key:
            companies = companies.drop_duplicates(lookup_key)

            jobs = jobs.merge(
                companies,
                left_on=company_key,
                right_on=lookup_key,
                how="left",
                suffixes=("", "_company"),
                validate="many_to_one"
            )

    # 3. Salaries: preserve separate salary records if multiple
    salaries = load_csv("salaries.csv")

    if salaries is not None:
        salary_job_key = find_column(salaries, ["job_id"])

        if salary_job_key and salary_job_key in salaries.columns:
            if salary_job_key == job_key:
                pass

            elif not salaries[salary_job_key].duplicated().any():
                jobs = jobs.merge(
                    salaries,
                    left_on=job_key,
                    right_on=salary_job_key,
                    how="left",
                    suffixes=("", "_salary"),
                    validate="one_to_one"
                )
            else:
                print(
                    "Multiple salary records per job detected. "
                    "Keeping salary data in a separate file "
                    "to avoid duplicating job rows."
                )

                salaries.to_csv(
                    os.path.join(DATA_DIR, "salary_records.csv"),
                    index=False
                )

    # 4. Skills: map skill IDs to readable names
    job_skills = load_csv("job_skills.csv")
    skills = load_csv("skills.csv")

    if job_skills is not None:
        skill_key = find_column(job_skills, ["skill_abr", "skill_id"])
        job_skill_key = find_column(job_skills, ["job_id"])

        if skills is not None and skill_key:
            lookup_skill_key = find_column(
                skills, ["skill_abr", "skill_id"]
            )
            skill_name = find_column(
                skills, ["skill_name", "name"]
            )

            if lookup_skill_key and skill_name:
                job_skills = job_skills.merge(
                    skills[[lookup_skill_key, skill_name]].drop_duplicates(),
                    left_on=skill_key,
                    right_on=lookup_skill_key,
                    how="left",
                    suffixes=("", "_lookup")
                )

                value = skill_name
            else:
                value = skill_key
        else:
            value = skill_key

        if job_skill_key and value:
            result = aggregate_values(
                job_skills, job_skill_key, value, "job_skills"
            )

            if result is not None:
                jobs = jobs.merge(
                    result, left_on=job_key, right_on=job_skill_key,
                    how="left", validate="one_to_one"
                )

    # 5. Industries associated with jobs
    job_industries = load_csv("job_industries.csv")
    industries = load_csv("industries.csv")

    if job_industries is not None:
        ji_job_key = find_column(job_industries, ["job_id"])
        ji_industry_key = find_column(
            job_industries, ["industry_id", "industry"]
        )

        value = ji_industry_key

        if industries is not None and ji_industry_key:
            lookup_industry_key = find_column(
                industries, ["industry_id", "id"]
            )
            industry_name = find_column(
                industries, ["industry_name", "name", "industry"]
            )

            if lookup_industry_key and industry_name:
                job_industries = job_industries.merge(
                    industries[[lookup_industry_key, industry_name]].drop_duplicates(),
                    left_on=ji_industry_key,
                    right_on=lookup_industry_key,
                    how="left",
                    suffixes=("", "_lookup")
                )
                value = industry_name

        if ji_job_key and value:
            result = aggregate_values(
                job_industries, ji_job_key, value, "job_industries"
            )

            if result is not None:
                jobs = jobs.merge(
                    result, left_on=job_key, right_on=ji_job_key,
                    how="left", validate="one_to_one"
                )

    # 6. Benefits
    benefits = load_csv("benefits.csv")

    if benefits is not None:
        benefit_job_key = find_column(benefits, ["job_id"])
        benefit_value = find_column(benefits, ["type", "benefit"])

        if benefit_job_key and benefit_value:
            result = aggregate_values(
                benefits, benefit_job_key, benefit_value, "benefits"
            )

            if result is not None:
                jobs = jobs.merge(
                    result, left_on=job_key, right_on=benefit_job_key,
                    how="left", validate="one_to_one"
                )

    # 7. Company industries and specialties
    company_key = find_column(jobs, ["company_id"])

    for filename, candidates, output_name in [
        ("company_industries.csv", ["industry", "industry_name"], "company_industries"),
        ("company_specialities.csv", ["speciality", "specialty"], "company_specialities"),
    ]:
        table = load_csv(filename)

        if table is None or company_key is None:
            continue

        table_company_key = find_column(table, ["company_id"])
        value_column = find_column(table, candidates)

        if table_company_key and value_column:
            result = aggregate_values(
                table, table_company_key, value_column, output_name
            )

            if result is not None:
                result = result.rename(
                    columns={table_company_key: company_key}
                )

                jobs = jobs.merge(
                    result, on=company_key, how="left",
                    validate="many_to_one"
                )

    # 8. Save the combined dataset
    output_path = os.path.join(
        DATA_DIR, "merged_linkedin_jobs.csv"
    )

    jobs.to_csv(output_path, index=False)

    print("\nMERGE COMPLETE")
    print(f"Output: {output_path}")
    print(f"Jobs: {len(jobs):,}")
    print(f"Columns: {len(jobs.columns):,}")


if __name__ == "__main__":
    main()
