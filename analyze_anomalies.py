import json
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path("C:/Users/Gopika Arasi/OneDrive/Documents/Desktop/indiarun/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge")
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None

def analyze_dataset():
    anomaly_categories = Counter()
    specific_examples = {}
    
    founding_dates = {
        "openai": 2015,
        "snowflake": 2012,
        "stripe": 2010,
        "zoom": 2011,
        "tiktok": 2016,
        "slack": 2009,
        "figma": 2012,
        "databricks": 2013,
        "confluent": 2014,
        "hashicorp": 2012,
        "anthropic": 2021,
        "cohere": 2019,
        "pinecone": 2019,
        "weaviate": 2016,
        "qdrant": 2021,
    }
    
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            profile = cand.get("profile", {})
            history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            edu = cand.get("education", [])
            
            # Check 1: Working at a company before its founding date
            for job in history:
                comp = job.get("company", "").lower().strip()
                start = parse_date(job.get("start_date"))
                if start and comp:
                    for fc, year in founding_dates.items():
                        if fc in comp and start.year < year:
                            cat = "Company founding date mismatch"
                            anomaly_categories[cat] += 1
                            if cat not in specific_examples:
                                specific_examples[cat] = (cid, f"Worked at {job['company']} in {start.year} (founded in {year})")
                                
            # Check 2: Skill inflation (expert skills with 0 duration)
            expert_zero_dur = 0
            for s in skills:
                if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0:
                    expert_zero_dur += 1
            if expert_zero_dur >= 5:
                cat = f"Skill inflation (>=5 expert skills with 0 duration)"
                anomaly_categories[cat] += 1
                if cat not in specific_examples:
                    specific_examples[cat] = (cid, f"{expert_zero_dur} expert skills with 0 duration")
                    
            # Check 3: Duration months vs date diff mismatch (large mismatch)
            for i, job in enumerate(history):
                start = parse_date(job.get("start_date"))
                end = parse_date(job.get("end_date"))
                duration_months = job.get("duration_months", 0)
                if start:
                    actual_end = end if end else datetime(2026, 6, 24)
                    diff_months = (actual_end.year - start.year) * 12 + (actual_end.month - start.month)
                    if abs(duration_months - diff_months) > 24: # Mismatch of more than 2 years
                        cat = "Job duration vs dates mismatch (>24 months)"
                        anomaly_categories[cat] += 1
                        if cat not in specific_examples:
                            specific_examples[cat] = (cid, f"Job {i} ({job['company']}) suggests {diff_months} months, but duration_months is {duration_months}")
                            
            # Check 4: Profile years of experience vs history sum mismatch
            sum_months = sum(job.get("duration_months", 0) for job in history)
            profile_years = profile.get("years_of_experience", 0)
            if abs(profile_years - (sum_months / 12.0)) > 5.0:
                cat = "Profile experience vs career history mismatch (>5 years)"
                anomaly_categories[cat] += 1
                if cat not in specific_examples:
                    specific_examples[cat] = (cid, f"Profile says {profile_years} yrs, but history says {sum_months/12.0:.1f} yrs")
                    
            # Check 5: Career started before education started
            if edu and history:
                min_edu_start = min((e.get("start_year") for e in edu if e.get("start_year")), default=None)
                min_career_start = min((parse_date(j.get("start_date")).year for j in history if parse_date(j.get("start_date"))), default=None)
                if min_edu_start and min_career_start and min_career_start < min_edu_start - 2:
                    cat = "Career started before education started"
                    anomaly_categories[cat] += 1
                    if cat not in specific_examples:
                        specific_examples[cat] = (cid, f"Career started {min_career_start}, edu started {min_edu_start}")
                        
            # Check 6: Multiple current jobs
            current_jobs = [j for j in history if j.get("is_current")]
            if len(current_jobs) > 1:
                cat = "Multiple current jobs"
                anomaly_categories[cat] += 1
                if cat not in specific_examples:
                    specific_examples[cat] = (cid, f"Jobs at {[j.get('company') for j in current_jobs]}")
                    
    print("--- ANOMALY CATEGORY COUNTS ---")
    for cat, count in anomaly_categories.items():
        print(f"{cat}: {count}")
        if cat in specific_examples:
            cid, detail = specific_examples[cat]
            print(f"  Example: {cid} - {detail}")
            
analyze_dataset()
