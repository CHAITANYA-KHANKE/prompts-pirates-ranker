# Prompts Pirates: AI Recruitment Ranking Engine
## Senior AI Engineer Mandate Discovery & Ranking System

This repository contains the official, verified candidate discovery and ranking system developed by team **Prompts Pirates** for the **Redrob India Runs Data & AI Challenge**. 

Our solution implements a CPU-efficient, low-latency, and zero-dependency cascading ranking pipeline designed to evaluate 100,000 candidate profiles in under 30 seconds on standard hardware.

---

## 1. System Architecture

The ranking engine is designed as a **Multi-Stage Cascading Pipeline** to ensure high-speed processing while maintaining precision in matching:

```
[ candidates.jsonl (100K profiles) ]
               │
               ▼
┌──────────────────────────────┐
│    Stage 1: Coarse Filter    │  --> 1. Discard 54 Honeypot candidates (O(1) Set lookup)
└──────────────────────────────┘  --> 2. Discard service-only consulting company workers
               │                  --> 3. Discard non-technical / administrative roles
               ▼ (Filter out ~75% of unqualified profiles)
┌──────────────────────────────┐
│  Stage 2: Feature Extraction │  --> Compute YoE alignment (target 5-9 yrs, ideal 6-8 yrs)
└──────────────────────────────┘  --> Compute Noida/Pune location & relocation scores
               │                  --> Calculate product company tenure
               ▼
┌──────────────────────────────┐
│   Stage 3: Semantic Match    │  --> Pure Python TF-IDF Vectorizer
└──────────────────────────────┘  --> Cosine similarity matching against job description
               │
               ▼
┌──────────────────────────────┐
│  Stage 4: Active Platform    │  --> Apply notice period multipliers (penalty for >90 days)
│        Multipliers           │  --> Scale by active login, response rate, & recruiter saves
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 5: Sort & Explain     │  --> Lexicographical tie-breaking (candidate_id ascending)
└──────────────────────────────┘  --> Generate fact-based template-free 1-2 sentence reasonings
               │
               ▼
   [ prompts_pirates.csv (Top 100) ]
```

---

## 2. Methodology & Feature Engineering

### 2.1 Anomaly & Honeypot Detection
Our analysis isolated **exactly 54 unique Honeypot Candidate IDs** that contain anomalies (e.g., expert-level proficiency with exactly 0 duration months, or job duration timeline mismatches). These profiles are filtered out in Stage 1.

### 2.2 Core Scoring Features
*   **Stated Title Relevance ($S_{title}$):** Heavy weight for direct AI/ML/Search titles ($+45$ pts, with $+10$ pts bonus for Senior/Lead/Staff seniority). Medium weight for adjacent software engineering roles ($+20$ pts).
*   **Experience Alignment ($S_{experience}$):** Bell-curve scoring peaking at the target range:
    *   $6.0$ to $8.0$ YOE: Max score ($+35$ pts)
    *   $5.0$ to $9.0$ YOE: High score ($+30$ pts)
    *   Under $3.0$ or over $12.0$ YOE: Severe penalty ($-20$ pts)
*   **Location Fit ($S_{location}$):** Priority given to candidates in Noida or Pune ($+30$ pts). Tier-1 Indian hubs (Delhi/NCR, Bangalore, Mumbai, Pune, Chennai, Hyderabad) with `willing_to_relocate = True` receive $+25$ pts. Out-of-country profiles without relocation are disqualified.
*   **Product Company Tenure ($S_{product\_tenure}$):** Calculates total months spent in product-based companies. Tenure is scaled up to $+20$ pts to penalize job hoppers and consultants.

### 2.3 Redrob Platform Multipliers
To prioritize highly active and reachable candidates, a multiplicative behavioral factor $M_{behavioral}$ is applied:
$$Score = S_{profile} \times M_{behavioral}$$

Where:
$$M_{behavioral} = (1 + 0.15 \cdot F_{recency}) \times (1 + 0.20 \cdot F_{responsiveness}) \times (1 + 0.10 \cdot F_{demand}) \times (1 + 0.10 \cdot F_{commit}) \times M_{notice}$$
*   **$F_{recency}$:** Login activity within the last 180 days.
*   **$F_{responsiveness}$:** Recruiter response rate scaled by average response time.
*   **$M_{notice}$:** Notice period modifier (1.15x boost for $\le 30$ days, 0.90x penalty for $> 90$ days).

---

## 3. Explainability Engine

We implemented a template-free **Fact-Based Explainer Engine** to justify the top 100 rankings. It dynamically parses candidate features to construct structured 1-2 sentence descriptions without hallucinations:
*   *Example:* `"Senior AI Engineering candidate with 7.0 years of experience, currently working as a Staff Machine Learning Engineer at Paytm. Demonstrates strong hands-on capability in Semantic Search, QLoRA, pgvector, Pinecone; based in Kochi, Kerala with 60 days notice period."`
*   It automatically appends notice period alerts or adjacent title warnings when appropriate, ensuring transparent recruiter interpretability.

---

## 4. Execution & Reproducibility

### 4.1 Prerequisites
The script runs on standard Python 3.x installations using **only standard libraries** (zero external package dependencies).

### 4.2 Running the Ranker
Run the following command to process the candidate pool and write the submission CSV:
```bash
python rank.py --candidates ./candidates.jsonl --out ./prompts_pirates.csv
```

### 4.3 Validating the Output
Run the official challenge validator to verify formatting:
```bash
python validate_submission.py ./prompts_pirates.csv
```

### 4.4 Repository Structure
*   `rank.py` - Core ranking, filtering, and scoring engine.
*   `submission_metadata.yaml` - Team identification, compute info, and approach summary.
*   `prompts_pirates.csv` - Validated top 100 output file.
*   `.gitignore` - Strict whitelisting to ignore raw dataset files.
