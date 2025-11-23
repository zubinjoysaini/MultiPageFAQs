# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 12:06:33 2025

@author: zubin
"""

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os, re, time, io
import requests
import pandas as pd
from serpapi.google_search import GoogleSearch

# -----------------------
# Load keys
# -----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()
if not SERPAPI_KEY:
    st.error("SERPAPI_KEY missing in .env (get one from https://serpapi.com)")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# Helpers
# -----------------------
def serp_search(query, serpapi_key, num_results=3, time_period=None):
    """
    Query Google via SerpApi.
    time_period: optional (e.g., 'd' for past day, 'w' for week) - SerpApi supports advanced params; omitted here.
    Returns list of dicts: {title, link, snippet, date}
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key,
        "num": num_results,
        # you can add additional params like 'gl','hl','google_domain' etc.
    }
    # If you want to restrict by time, SerpApi supports 'tbs' param, e.g. 'qdr:w' for week
    # Example: params["tbs"] = "qdr:w"  # last week
    search = GoogleSearch(params)
    res = search.get_dict()
    results = []
    for item in res.get("organic_results", [])[:num_results]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "") or ""
        date = item.get("date") or item.get("published") or ""
        results.append({"title": title, "link": link, "snippet": snippet, "date": date})
    return results

def url_is_alive(url, timeout=6):
    """Simple URL check: HEAD then GET fallback."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code >= 400:
            r = requests.get(url, allow_redirects=True, timeout=timeout)
        return r.status_code < 400
    except Exception:
        return False

def sanitize_sheet_name(name):
    """Excel sheet name must be <=31 chars and not contain certain characters"""
    safe = re.sub(r"[\\/*?:\[\]]", "_", name)[:31]
    return safe or "sheet"

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="SerpApi Grounded Multi-Topic Q&A", layout="wide")
st.title("🔎 Multi-Topic Grounded Q&A (SerpApi)")

st.markdown(
    "Enter multiple topics (comma- or newline-separated). For each topic the app will:\n"
    "1. Collate real search snippets via SerpApi, validate links,\n"
    "2. Ask the model to **answer using only the provided verified links**,\n"
    "3. Provide per-topic Excel download and a combined workbook."
)

# Inputs
col1, col2 = st.columns([3,1])
with col1:
    topic_input = st.text_area("Enter topics (comma or newline separated):", height=140,
                               placeholder="e.g. electric vehicles\nai in education, kabaddi leagues")
with col2:
    num_questions = st.number_input("Top questions per topic", min_value=1, max_value=30, value=15, step=1)
    results_per_question = st.number_input("Search results per question", min_value=1, max_value=8, value=3, step=1)
    freshness_example = st.selectbox("Only recent sources (optional)", options=["None","Past day","Past week","Past month"], index=0)

generate = st.button("Generate Q&A for all topics")

# Map freshness selection to tbs param (optional)
tbs_map = {"None": None, "Past day": "qdr:d", "Past week": "qdr:w", "Past month": "qdr:m"}

if generate:
    if not topic_input.strip():
        st.warning("Please enter at least one topic.")
        st.stop()

    topics = [t.strip() for t in re.split(r",|\n", topic_input) if t.strip()]
    st.info(f"Processing {len(topics)} topics — this may take several minutes depending on counts and rate limits.")

    all_buffers = {}
    overall_progress = st.progress(0)
    total_steps = len(topics)
    step_idx = 0

    for topic in topics:
        step_idx += 1
        st.header(f"Topic: {topic}")
        st.write("Collecting question candidates from web search (serp)...")

        # --- Gather candidate question-like snippets by searching queries about the topic
        seed_queries = [
            f"most asked ChatGPT questions about {topic}",
            f"popular prompts about {topic}",
            f"frequently asked questions {topic}",
            f"people also ask {topic}",
            f"common questions about {topic}"
        ]
        snippets = []
        for q in seed_queries:
            try:
                # include tbs param if freshness requested
                params_q = q
                search_res = serp_search(params_q, SERPAPI_KEY, num_results=5)
                for item in search_res:
                    text = (item.get("title") or "") + " " + (item.get("snippet") or "")
                    if text.strip():
                        snippets.append(text)
                time.sleep(0.4)  # polite pacing
            except Exception as e:
                st.warning(f"Seed search failed for '{q}': {e}")

        if not snippets:
            st.warning(f"No search snippets found for '{topic}'. Skipping.")
            overall_progress.progress(step_idx/total_steps)
            continue

        combined_snippets = "\n".join(snippets)

        # --- Use OpenAI to extract the top N real questions from snippets
        extract_prompt = (
            f"The text below contains headlines and snippets related to '{topic}'.\n"
            f"From this text, extract the top {int(num_questions)} most common, realistic user-style questions "
            "people are actually asking (e.g., how-to, comparisons, troubleshooting). "
            "Return them as a clear numbered list only.\n\n"
            f"Text:\n{combined_snippets}"
        )
        try:
            q_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are an assistant that extracts frequent user questions."},
                          {"role":"user","content":extract_prompt}]
            )
            raw_q_text = q_resp.choices[0].message.content.strip()
            candidate_qs = [re.sub(r"^[0-9\.\)\-\s]+","",line).strip() for line in raw_q_text.splitlines() if line.strip()]
            questions = candidate_qs[:int(num_questions)]
        except Exception as e:
            st.error(f"Error extracting questions via OpenAI: {e}")
            questions = []

        if not questions:
            st.warning(f"No questions extracted for '{topic}'. Skipping.")
            overall_progress.progress(step_idx/total_steps)
            continue

        st.subheader("Top questions found")
        for i, qq in enumerate(questions, 1):
            st.write(f"{i}. {qq}")

        qa_rows = []
        q_progress = st.progress(0)

        # For each question: fetch live search results, validate links, ground LLM
        for i, question in enumerate(questions):
            st.markdown(f"---\n**Q{i+1}. {question}**")
            # Fetch search results for this question
            try:
                # append year filter to encourage recent results optionally
                search_query = f"{question}"
                search_results = serp_search(search_query, SERPAPI_KEY, num_results=int(results_per_question))
            except Exception as e:
                st.warning(f"Search failed for question: {e}")
                search_results = []

            # Validate results: keep only reachable URLs
            verified = []
            for r in search_results:
                link = r.get("link")
                if link and url_is_alive(link):
                    verified.append(r)
                # small delay to avoid hammering
                time.sleep(0.2)

            # If none validated, allow unvalidated results as fallback (you can change policy)
            used_results = verified if verified else search_results

            # Build grounding context text
            if used_results:
                grounding_lines = []
                for r in used_results:
                    title = r.get("title","")
                    link = r.get("link","")
                    snippet = r.get("snippet","").replace("\n"," ")
                    grounding_lines.append(f"{title} | {link} | {snippet}")
                grounding_text = "\n".join(grounding_lines)
            else:
                grounding_text = ""

            # Compose answer prompt - instruct to use ONLY provided links
            if grounding_text:
                answer_prompt = (
                    "You are given a question and a list of verified web results (title | url | snippet).\n"
                    "CONSTRAINT: Use ONLY the provided URLs and snippets to form your answer. Do NOT invent or hallucinate sources.\n"
                    "If the sources do not provide enough information to answer, respond with: 'Insufficient evidence in the provided sources.'\n\n"
                    f"Question:\n{question}\n\n"
                    f"Search results (TITLE | URL | SNIPPET):\n{grounding_text}\n\n"
                    "Produce:\n1) A concise answer (2-6 sentences).\n2) A 'Sources:' section listing the URLs you used (subset of the provided URLs).\n"
                )
            else:
                answer_prompt = (
                    f"Question:\n{question}\n\n"
                    "No verified search results were found. State 'No verified sources found' and provide no fabricated URLs.\n"
                )

            # Call OpenAI to synthesize an answer grounded on real links
            try:
                a_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"You are a factual assistant that cites only provided sources."},
                              {"role":"user","content":answer_prompt}]
                )
                answer_text = a_resp.choices[0].message.content.strip()
            except Exception as e:
                answer_text = f"Error generating answer: {e}"

            # Try to extract 'Sources:' block from answer_text
            sources_text = ""
            m = re.search(r"(Sources|References)[:\s]*(.*)$", answer_text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                sources_text = m.group(2).strip()
            else:
                # fallback: list the verified URLs used
                sources_text = "; ".join([r.get("link","") for r in used_results]) if used_results else ""

            # Display
            st.write(answer_text)
            if used_results:
                st.markdown("**Verified links used:**")
                for r in used_results:
                    st.write(r.get("link"))

            qa_rows.append({
                "Topic": topic,
                "Question": question,
                "Answer": answer_text,
                "Verified_Links": "; ".join([r.get("link","") for r in used_results]),
                "Search_Snippets": " || ".join([r.get("snippet","") for r in used_results])
            })

            q_progress.progress((i+1)/len(questions))
            time.sleep(0.5)  # pacing

        # Save this topic's QA to an in-memory Excel file
        df = pd.DataFrame(qa_rows)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            sheet_name = sanitize_sheet_name(topic)
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        buffer.seek(0)
        all_buffers = locals().get("all_buffers", {})
        all_buffers[topic] = buffer

        # Provide download for this topic
        st.download_button(
            label=f"Download Excel for '{topic}'",
            data=buffer,
            file_name=f"{topic.replace(' ','_')}_QA.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        overall_progress.progress(step_idx/total_steps)

    # Optionally create a combined workbook containing separate sheets per topic
    if 'all_buffers' in locals() and all_buffers:
        combined_buf = io.BytesIO()
        with pd.ExcelWriter(combined_buf, engine="openpyxl") as writer:
            for topic_name, buf in all_buffers.items():
                buf.seek(0)
                df = pd.read_excel(buf)
                sheet_name = sanitize_sheet_name(topic_name)
                # If sheet name conflict, pandas/openpyxl will handle by truncation; ensure uniqueness if needed
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        combined_buf.seek(0)
        st.download_button(
            label="Download Combined Workbook (All Topics)",
            data=combined_buf,
            file_name="all_topics_QA_workbook.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


    st.success("All done — files ready for download.")
