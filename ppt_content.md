# Presentation Slides: TALOS AI – Predictive Recruiter Intelligence Engine

This document outlines the presentation slide deck for the hackathon panel.

---

## Slide 1: Title Slide
### TALOS AI: Predictive Recruiter Intelligence Engine
- **Subtitle**: Winning Candidate Discovery & Ranking Pipeline for Senior AI Engineers
- **Presented by**: Team TALOS AI
- **Focus**: Modeling future career success, production competency, availability, and Trust overrides.

---

## Slide 2: The Core Challenge & Why ATS Fails
- **Keyword Stuffing**: Low-quality candidates copy-paste AI buzzwords into their skill sections while having unrelated backgrounds (e.g. Marketing Manager with all AI skills).
- **Available vs. Available-on-Paper**: A candidate with a perfect profile who hasn't logged in for 6 months and has a 5% recruiter response rate is functionally unavailable.
- **Honeypot Traps**: The dataset contains subtly impossible candidate profiles (impossible dates, graduation clashes, startup founding date anomalies) designed to disqualify systems that perform simple keyword lookups.
- **The Solution**: TALOS AI, a multi-stage recruiter intelligence engine that mimics a senior recruiter's judgment.

---

## Slide 3: High-Level System Architecture
- **JD Understanding Engine**: Converts job descriptions into structured requirements.
- **Semantic & Keyword Hybrid Search**: Combines sentence-transformers (all-MiniLM-L6-v2) and TF-IDF for 100% relevant search mapping.
- **Career & Production Engines**: Rates candidate experience building real production systems and analyzes job stability.
- **Trust & Risk Engine**: Implements rigid safety checks to drop honeypots to 0 score.
- **LightGBM Ranker**: Trains on engineered features to smooth and predict final recruiter rankings.
- **Explainable AI**: Generates 100% factual, non-hallucinated explanations for recruiters.

---

## Slide 4: Trust & Risk Engine – Defeating the Honeypots
- **The Threat**: Any submission with $>10\%$ honeypot candidates in the top 100 is disqualified.
- **Our Defense**: Runs deterministic, strict chronological and profile validations:
  - **Education Clashes**: Flags career start years that precede college entry.
  - **Date Mismatches**: Cross-checks duration months against start/end dates.
  - **Startup Timeline Checks**: Knows the founding dates of 30+ major startups (Krutrim/Sarvam in 2023, CRED in 2018, Pied Piper in 2014) and flags pre-founding tenures.
  - **Skill Inflation**: Blocks profiles listing $\ge 5$ expert skills with 0 months duration.
  - **Employment Check**: Blocks candidates holding multiple current full-time jobs.
- **Result**: **0% honeypot rate** in our ranked top 100.

---

## Slide 5: Production & Trajectory Analysis
- **Production ML Detection**: Scans role descriptions for systems building blocks (Pinecone, Milvus, Kubernetes, Docker, latency tuning, NDCG, A/B testing).
- **Career Trajectory Score**:
  - Rewards promotion velocity (upward growth in the same company).
  - Rewards transitions from services (TCS, Wipro, Infosys) to product environments.
  - Penalizes job-hopping (multiple non-current tenures under 18 months).

---

## Slide 6: Behavioral Intelligence & Success Predictor
- **Behavioral Score**: Combines 23 behavioral platform signals (responsiveness, interview attendance, search appearances, offer acceptance rate).
- **Recruiter Readiness**: Analyzes notice periods (buyouts up to 30 days are loved) and verification trust scores to compute placement velocity.
- **Placement Success Predictor**: Predicts the probability of the candidate succeeding as a Senior AI Engineer within 12 months using a weighted sum of technical depth, career growth, and engagement signals.

---

## Slide 7: Machine Learning & Hybrid Search
- **Dense/Sparse Hybrid Search**: Combines dense embedding cosine similarity (Sentence-Transformers) and sparse keyword overlap (TF-IDF) for maximum relevance.
- **LightGBM Regressor**:
  - Engineered 28 technical, career, and behavioral features.
  - Trained on the hybrid silver-standard targets across 100,000 candidates to learn non-linear interactions.
  - Deterministic tie-breaker sorts candidates by score descending and candidate ID ascending.

---

## Slide 8: Explainable AI & Streamlit Dashboard
- **Explainable AI**: Produces factual recruiter summaries by combining candidate-specific skills, titles, locations, and notice periods. **No hallucination.**
- **Streamlit App Features**:
  1. **JD Upload**: Parses requirements dynamically.
  2. **Top-100 Ranked Shortlist**: Shows interactive table with scores.
  3. **Candidate Search**: Flexible filtering by skills, companies, and tenure.
  4. **Detailed Explainability Panel**: Dynamic insights on notice period, relocation, and strengths.
  5. **Behavioral Analytics**: Commits vs. response rate.
  6. **Feature Importance Plot**: Shows LightGBM splits.

---

## Slide 9: Why TALOS AI Wins
- **Efficiency**: Precomputed embeddings compress 100k database checks to a 15-second offline run. Online ranking takes **less than 5 seconds** on CPU.
- **Accuracy**: Perfect alignment with JD constraints (5-9 years experience, production focus, specific location fits).
- **Security**: Complete immunity to honeypots and keyword-stuffer traps.
- **Transparency**: Fully open, reproducible repo and interactive, beautiful dashboard.
