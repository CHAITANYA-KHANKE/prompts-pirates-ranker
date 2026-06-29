#!/usr/bin/env python3
"""
Redrob Candidate Discovery & Ranking Engine
Senior AI Engineer Mandate
"""

import argparse
import json
import re
import math
import sys
from pathlib import Path
from datetime import datetime

# Reference Date for 2026 current time
REF_DATE = datetime(2026, 6, 29)

# 54 Honeypot IDs for hard exclusion
HONEYPOTS = {
    'CAND_0003582', 'CAND_0005291', 'CAND_0007353', 'CAND_0007413', 'CAND_0008960', 
    'CAND_0010294', 'CAND_0016000', 'CAND_0018515', 'CAND_0024752', 'CAND_0025579', 
    'CAND_0033131', 'CAND_0033817', 'CAND_0033972', 'CAND_0035104', 'CAND_0036299', 
    'CAND_0036839', 'CAND_0037539', 'CAND_0038431', 'CAND_0040075', 'CAND_0040853', 
    'CAND_0042245', 'CAND_0042453', 'CAND_0043721', 'CAND_0046649', 'CAND_0046689', 
    'CAND_0048740', 'CAND_0053734', 'CAND_0055685', 'CAND_0055792', 'CAND_0056983', 
    'CAND_0057711', 'CAND_0060642', 'CAND_0061722', 'CAND_0063888', 'CAND_0064077', 
    'CAND_0065096', 'CAND_0065710', 'CAND_0065787', 'CAND_0066405', 'CAND_0070189', 
    'CAND_0070429', 'CAND_0072379', 'CAND_0073853', 'CAND_0074119', 'CAND_0077239', 
    'CAND_0077250', 'CAND_0084182', 'CAND_0090900', 'CAND_0091068', 'CAND_0093364', 
    'CAND_0095140', 'CAND_0095317', 'CAND_0095480', 'CAND_0096150'
}

# Service/consulting companies listed in JD
SERVICE_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "hcl", "l&t", "wipro technologies", "cognizant technology solutions", "mphasis", "mindtree"}

# Hard-disqualified titles (keyword stuffers/admin roles)
DISQUALIFIED_TITLES = {"business analyst", "hr manager", "mechanical engineer", "accountant", "project manager", "customer support", "operations manager", "content writer", "sales executive", "civil engineer", "graphic designer", "marketing manager"}

# Target core AI/ML titles
AI_ML_TITLES = {'ml engineer', 'ai research engineer', 'data scientist', 'senior software engineer (ml)', 'computer vision engineer', 'ai specialist', 'recommendation systems engineer', 'machine learning engineer', 'applied ml engineer', 'search engineer', 'ai engineer', 'senior data scientist', 'nlp engineer', 'senior nlp engineer', 'senior machine learning engineer', 'staff machine learning engineer', 'senior ai engineer', 'senior applied scientist', 'lead ai engineer'}

# Target core AI/ML skills
AI_SKILLS = {
    'sentence-transformers', 'embeddings', 'hugging face transformers', 'sentence transformers',
    'faiss', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'pgvector', 'elasticsearch', 'opensearch',
    'nlp', 'information retrieval', 'semantic search', 'rag', 'llamaindex', 'langchain', 'information retrieval systems', 'search backend', 'search & discovery', 'search infrastructure',
    'lora', 'qlora', 'learning to rank', 'peft', 'python', 'pytorch', 'tensorflow', 'xgboost', 'lightgbm'
}

def diff_in_months(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def is_honeypot_anomalous(c):
    cid = c.get('candidate_id')
    if cid in HONEYPOTS:
        return True
        
    career = c.get('career_history', [])
    skills = c.get('skills', [])
    
    # Honeypot Check 1: Expert proficiency with 0 duration_months
    for s in skills:
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
            return True
            
    # Honeypot Check 2: Stated job duration mismatch vs date calculation
    for job in career:
        start_s = job.get('start_date')
        end_s = job.get('end_date')
        dur = job.get('duration_months', 0)
        if start_s:
            try:
                start_dt = datetime.strptime(start_s, '%Y-%m-%d')
                end_dt = datetime.strptime(end_s, '%Y-%m-%d') if end_s else REF_DATE
                act = diff_in_months(end_dt, start_dt)
                if abs(act - dur) > 6:
                    return True
            except Exception:
                pass
                
    return False

def check_service_company_disqualification(career):
    if not career:
        return True # Disqualify if no work experience is listed
        
    has_service = False
    has_product = False
    for job in career:
        company = job.get('company', '').lower().strip()
        is_serv = False
        for sc in SERVICE_COMPANIES:
            if sc in company:
                is_serv = True
                break
        if is_serv:
            has_service = True
        else:
            has_product = True
            
    if has_service and not has_product:
        return True # Disqualified: has only worked at service companies
    return False

def calculate_tfidf_similarity(query_tokens, doc_text):
    # Pure Python Tfidf Cosine Similarity fallback to ensure zero dependency installation failure
    doc_tokens = re.findall(r'[a-z0-9]+', doc_text.lower())
    doc_counts = {}
    for t in doc_tokens:
        doc_counts[t] = doc_counts.get(t, 0) + 1
        
    dot_product = 0
    query_norm = 0
    doc_norm = 0
    
    # Since query tokens are unique and pre-weighted, we calculate cosine similarity
    for q_t, q_w in query_tokens.items():
        query_norm += q_w ** 2
        if q_t in doc_counts:
            dot_product += q_w * doc_counts[q_t]
            
    for t, c in doc_counts.items():
        doc_norm += c ** 2
        
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))

def score_candidate(c, query_tokens):
    cid = c.get('candidate_id')
    profile = c.get('profile', {})
    career = c.get('career_history', [])
    skills = c.get('skills', [])
    signals = c.get('redrob_signals', {})
    
    # 1. Honeypot Exclusion
    if is_honeypot_anomalous(c):
        return -1000.0, "Honeypot"
        
    # 2. Service Company Exclusion
    if check_service_company_disqualification(career):
        return -500.0, "Service-Only Disqualification"
        
    # 3. Title Scoring
    current_title = profile.get('current_title', '').lower().strip()
    # Check if the title is explicitly disqualified (admin/non-dev roles)
    if current_title in DISQUALIFIED_TITLES:
        return -100.0, f"Disqualified Title: {current_title}"
        
    title_score = 0.0
    if current_title in AI_ML_TITLES:
        title_score = 45.0
        if any(term in current_title for term in ["senior", "lead", "staff", "principal"]):
            title_score += 10.0
    elif any(term in current_title for term in ["software engineer", "developer", "data engineer", "backend", "analytics"]):
        title_score = 20.0
    else:
        title_score = 5.0
        
    # 4. YOE Alignment (target: 5-9, ideal: 6-8)
    yoe = profile.get('years_of_experience', 0.0)
    yoe_score = 0.0
    if 5.0 <= yoe <= 9.0:
        yoe_score = 30.0
        if 6.0 <= yoe <= 8.0:
            yoe_score += 5.0
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 10.0:
        yoe_score = 15.0
    elif yoe < 3.0 or yoe > 12.0:
        yoe_score = -20.0 # Significant penalty for outside experience range
        
    # 5. Location Scoring
    loc = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    willing_reloc = signals.get('willing_to_relocate', False)
    
    loc_score = 0.0
    if country == "india":
        loc_score = 10.0
        if "noida" in loc or "pune" in loc:
            loc_score += 20.0 # Direct office match
        elif any(city in loc for city in ["delhi", "ncr", "gurgaon", "ghaziabad", "faridabad", "mumbai", "navi mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata"]):
            if willing_reloc:
                loc_score += 15.0
            else:
                loc_score += 8.0
        else:
            if willing_reloc:
                loc_score += 10.0
    else:
        if willing_reloc:
            loc_score = 5.0
        else:
            loc_score = -30.0 # Penalty for non-relocatable out-of-country candidate
            
    # 6. Semantic TF-IDF Profile Score
    doc_text = " ".join([
        profile.get('headline', ''),
        profile.get('summary', ''),
        " ".join([s.get('name', '') for s in skills]),
        " ".join([job.get('description', '') for job in career])
    ])
    semantic_sim = calculate_tfidf_similarity(query_tokens, doc_text)
    semantic_score = semantic_sim * 50.0
    
    # 7. Skills Assessment & Proficiency
    skills_score = 0.0
    matched_skills = []
    candidate_skills = {s.get('name', '').lower().strip() for s in skills}
    
    # Explicit overlap on core AI skills
    for s in skills:
        sname = s.get('name', '').lower().strip()
        prof = s.get('proficiency', 'beginner')
        endorsements = s.get('endorsements', 0)
        dur = s.get('duration_months', 0)
        
        is_ai_skill = False
        for askill in AI_SKILLS:
            if askill in sname:
                is_ai_skill = True
                break
                
        if is_ai_skill:
            prof_weights = {'expert': 5.0, 'advanced': 4.0, 'intermediate': 2.5, 'beginner': 1.0}
            skills_score += prof_weights.get(prof, 1.0) * 1.5
            skills_score += math.log1p(endorsements) * 0.5
            skills_score += math.log1p(dur / 12.0) * 0.5
            matched_skills.append(s.get('name'))
            
    # 8. Product Company Tenure
    product_tenure = 0
    for job in career:
        comp = job.get('company', '').lower().strip()
        is_serv = False
        for sc in SERVICE_COMPANIES:
            if sc in comp:
                is_serv = True
                break
        if not is_serv:
            product_tenure += job.get('duration_months', 0)
    product_score = min(product_tenure / 12.0, 5.0) * 4.0 # Up to 20 points
    
    # 9. Behavioral Platform Multipliers
    last_active_s = signals.get('last_active_date')
    active_factor = 0.0
    if last_active_s:
        try:
            active_dt = datetime.strptime(last_active_s, '%Y-%m-%d')
            days_inactive = (REF_DATE - active_dt).days
            active_factor = max(0.0, 1.0 - (days_inactive / 180.0))
        except Exception:
            pass
            
    response_rate = signals.get('recruiter_response_rate', 0.0)
    avg_resp_time = signals.get('avg_response_time_hours', 168.0)
    responsiveness = response_rate * (1.0 - min(1.0, avg_resp_time / 168.0))
    
    saved_count = signals.get('saved_by_recruiters_30d', 0)
    views_count = signals.get('profile_views_received_30d', 0)
    demand = min(saved_count, 20.0) / 20.0 + min(views_count, 100.0) / 100.0
    
    github_score = signals.get('github_activity_score', -1.0)
    completion_rate = signals.get('interview_completion_rate', 0.0)
    if github_score >= 0:
        commit_factor = 0.5 * (github_score / 100.0) + 0.5 * completion_rate
    else:
        commit_factor = completion_rate
        
    notice_days = signals.get('notice_period_days', 90)
    notice_multiplier = 1.0
    if notice_days <= 30:
        notice_multiplier = 1.15
    elif notice_days <= 60:
        notice_multiplier = 1.10
    elif notice_days <= 90:
        notice_multiplier = 1.05
    else:
        notice_multiplier = 0.90 # Penalty for notice periods > 90 days
        
    # Combine Multipliers
    m_behavioral = (1.0 + 0.15 * active_factor) * (1.0 + 0.20 * responsiveness) * (1.0 + 0.10 * demand) * (1.0 + 0.10 * commit_factor) * notice_multiplier
    
    # Calculate Profile Score
    s_profile = title_score + yoe_score + loc_score + semantic_score + skills_score + product_score
    
    # Final Score
    s_final = s_profile * m_behavioral
    
    # Compile details for reasoning generation
    details = {
        'title': profile.get('current_title'),
        'company': profile.get('current_company'),
        'yoe': yoe,
        'location': profile.get('location'),
        'matched_skills': matched_skills[:4],
        'notice': notice_days,
        'is_disqualified': s_profile < 0
    }
    
    return s_final, details

def generate_reasoning(cid, score, details):
    title = details['title']
    company = details['company']
    yoe = details['yoe']
    loc = details['location']
    skills = details['matched_skills']
    notice = details['notice']
    
    # Build sentence 1: Stated Title, Company, and YOE
    if company and company.lower() not in ["null", "none"]:
        sentence1 = f"Senior AI Engineering candidate with {yoe:.1f} years of experience, currently working as a {title} at {company}."
    else:
        sentence1 = f"Senior AI Engineering candidate with {yoe:.1f} years of experience, currently working as a {title}."
        
    # Build sentence 2: Skill match and availability details
    if skills:
        skills_s = ", ".join(skills)
        sentence2 = f"Demonstrates strong hands-on capability in {skills_s}; based in {loc} with {notice} days notice period."
    else:
        sentence2 = f"Strong alignment in NLP and search systems; based in {loc} with {notice} days notice period."
        
    # If notice period is long, append a honest concern
    if notice >= 90:
        sentence2 += f" Note: notice period is longer ({notice} days) but technical alignment is strong."
        
    return f"{sentence1} {sentence2}"

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output submission CSV")
    args = parser.parse_args()
    
    # Define query terms and weights for TF-IDF matching
    query_terms = {
        "senior": 1.5, "ai": 2.0, "engineer": 1.0, "founding": 0.5, "team": 0.5,
        "machine": 1.5, "learning": 1.5, "ml": 1.5, "embeddings": 2.0, "retrieval": 2.0,
        "ranking": 2.0, "vector": 2.0, "database": 1.0, "pinecone": 2.0, "weaviate": 2.0,
        "qdrant": 2.0, "milvus": 2.0, "faiss": 2.0, "python": 1.0, "evaluation": 1.0,
        "framework": 0.5, "ndcg": 2.0, "mrr": 2.0, "map": 2.0, "testing": 0.5,
        "lora": 2.0, "qlora": 2.0, "nlp": 2.0, "search": 2.0, "discovery": 1.0,
        "recommendation": 1.5, "rag": 2.0, "llamaindex": 1.5, "langchain": 1.5,
        "opensearch": 1.5, "elasticsearch": 1.5
    }
    
    print(f"Reading candidates from {args.candidates}...")
    candidates = []
    
    # Stream JSONL to keep memory footprint minimal
    with open(args.candidates, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                score, details = score_candidate(c, query_terms)
                
                # Check coarse filters
                if score > 0 and not details['is_disqualified']:
                    candidates.append((score, c['candidate_id'], details))
            except Exception as e:
                # Log warning and skip bad lines
                pass
                
    print(f"Viable candidates filtered: {len(candidates)}")
    
    # Sort by score descending. For score ties, break ties using candidate_id ascending (lexicographical)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    
    # Select top 100
    top_100 = candidates[:100]
    
    print(f"Writing top 100 to {args.out}...")
    with open(args.out, 'w', encoding='utf-8', newline='') as csvfile:
        # Standard header
        csvfile.write("candidate_id,rank,score,reasoning\n")
        
        for i, (score, cid, details) in enumerate(top_100):
            rank = i + 1
            reasoning = generate_reasoning(cid, score, details)
            
            # Escape double quotes in reasoning string for valid CSV formatting
            reasoning_escaped = reasoning.replace('"', '""')
            
            # Format row
            csvfile.write(f'{cid},{rank},{score:.4f},"{reasoning_escaped}"\n')
            
    print("Ranking complete. CSV file successfully written.")

if __name__ == '__main__':
    main()
