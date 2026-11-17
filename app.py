import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from database.db import (
    init_db, create_consultation, get_consultations, get_sections,
    add_feedback, save_prediction, get_feedback, get_section_id
)
from modules.sentiment import load_model, predict_sentiment
from modules.priority import detect_priority
from modules.keywords import extract_keywords

st.set_page_config(page_title="ConsultIQ", page_icon="📊", layout="wide")
init_db()

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)

st.sidebar.title("📊 ConsultIQ")
st.sidebar.caption("E-Consultation Feedback Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Consultations", "Submit Feedback", "Analytics", "Priority Review", "Feedback Explorer"]
)

consultations = get_consultations()

def analyze_and_store(feedback_id, comment):
    if model is None:
        raise RuntimeError("Model is not available. Train it using `python ml/train_model.py`.")
    sentiment, probabilities = predict_sentiment(comment, model)
    flagged, reason = detect_priority(comment, sentiment, probabilities)
    save_prediction(feedback_id, sentiment, probabilities, flagged, reason)
    return sentiment, probabilities, flagged, reason

def choose_consultation():
    if not consultations:
        st.info("Create a consultation first.")
        return None
    labels = {f'{c["title"]} (ID: {c["consultation_id"]})': c["consultation_id"] for c in consultations}
    selected = st.selectbox("Select consultation", list(labels.keys()))
    return labels[selected]

if model_error:
    st.error("The sentiment model has not been created yet.")
    st.code("python ml/train_model.py")
    st.caption(f"Details: {model_error}")

if page == "Overview":
    st.title("ConsultIQ — Overview")
    st.write("Analyse stakeholder feedback and identify recurring concerns from e-consultations.")
    records = get_feedback()
    total = len(records)
    labelled = [r for r in records if r["sentiment"]]
    priority_count = sum(int(r["is_priority"] or 0) for r in records)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total feedback", total)
    c2.metric("Analysed comments", len(labelled))
    c3.metric("Priority comments", priority_count)
    st.subheader("Recent feedback")
    if records:
        st.dataframe(pd.DataFrame(records)[
            ["consultation_title", "section_name", "comment", "sentiment", "is_priority"]
        ].head(8), use_container_width=True, hide_index=True)
    else:
        st.info("No feedback yet. Create a consultation and submit feedback to get started.")
    st.caption("Priority flags are intended to support administrator review, not confirm emergencies.")

elif page == "Consultations":
    st.title("Consultation Management")
    with st.form("create_consultation"):
        title = st.text_input("Consultation title", placeholder="Proposed City Infrastructure Improvements")
        description = st.text_area("Description")
        sections_text = st.text_input(
            "Sections (comma-separated)",
            placeholder="Road Infrastructure, Public Transport, Pedestrian Safety"
        )
        submitted = st.form_submit_button("Create consultation")
        if submitted:
            try:
                consultation_id = create_consultation(
                    title, description, sections_text.split(",")
                )
                st.success(f"Consultation created successfully. ID: {consultation_id}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Existing consultations")
    if consultations:
        for c in consultations:
            with st.expander(f'{c["title"]} — ID {c["consultation_id"]}'):
                st.write(c["description"] or "No description provided.")
                st.write("Sections:")
                for sec in get_sections(c["consultation_id"]):
                    st.write(f'• {sec["section_name"]}')
    else:
        st.info("No consultations have been created.")

elif page == "Submit Feedback":
    st.title("Feedback Collection")
    consultation_id = choose_consultation()
    if consultation_id is not None:
        sections = get_sections(consultation_id)
        section_names = [s["section_name"] for s in sections]
        if not section_names:
            st.warning("Add a section to this consultation first.")
        else:
            st.subheader("Submit an individual comment")
            with st.form("single_feedback"):
                section_name = st.selectbox("Consultation section", section_names)
                comment = st.text_area("Your feedback", height=120)
                send = st.form_submit_button("Submit and analyse")
                if send:
                    try:
                        if not comment.strip():
                            raise ValueError("Please enter a comment.")
                        feedback_id = add_feedback(
                            get_section_id(consultation_id, section_name), comment, "Manual"
                        )
                        sentiment, probs, flagged, reason = analyze_and_store(feedback_id, comment)
                        st.success(f"Feedback saved. Predicted sentiment: {sentiment}")
                        st.write("Prediction probabilities:")
                        st.json({k: round(v, 3) for k, v in probs.items()})
                        if flagged:
                            st.warning(f"Flagged for review: {reason}")
                    except Exception as exc:
                        st.error(str(exc))

            st.divider()
            st.subheader("Bulk import feedback from CSV")
            st.caption("Required columns: section, comment. Section names must match the selected consultation.")
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded is not None and st.button("Import and analyse CSV"):
                try:
                    df = pd.read_csv(uploaded)
                    if not {"section", "comment"}.issubset(df.columns):
                        raise ValueError("CSV must contain section and comment columns.")
                    imported, skipped = 0, 0
                    for _, row in df.iterrows():
                        section = str(row["section"]).strip()
                        comment = str(row["comment"]).strip()
                        section_id = get_section_id(consultation_id, section)
                        if not section_id or not comment or comment.lower() == "nan":
                            skipped += 1
                            continue
                        feedback_id = add_feedback(section_id, comment, "CSV")
                        analyze_and_store(feedback_id, comment)
                        imported += 1
                    st.success(f"Imported and analysed {imported} comments. Skipped {skipped} invalid or unmatched rows.")
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

elif page == "Analytics":
    st.title("Feedback Analytics")
    consultation_id = choose_consultation()
    if consultation_id is not None:
        records = get_feedback(consultation_id=consultation_id)
        df = pd.DataFrame(records)
        if df.empty:
            st.info("No feedback exists for this consultation yet.")
        else:
            labelled = df.dropna(subset=["sentiment"])
            a, b, c = st.columns(3)
            a.metric("Total comments", len(df))
            b.metric("Positive", int((labelled["sentiment"] == "Positive").sum()))
            c.metric("Negative", int((labelled["sentiment"] == "Negative").sum()))

            if not labelled.empty:
                sentiment_counts = labelled["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
                st.subheader("Overall sentiment distribution")
                st.plotly_chart(px.pie(
                    sentiment_counts, names="Sentiment", values="Count",
                    hole=0.4, title="Sentiment distribution"
                ), use_container_width=True)

                st.subheader("Section-wise sentiment")
                section_counts = labelled.groupby(["section_name", "sentiment"]).size().reset_index(name="Count")
                st.plotly_chart(px.bar(
                    section_counts, x="section_name", y="Count", color="sentiment",
                    barmode="group", title="Sentiment by consultation section"
                ), use_container_width=True)

            st.subheader("Frequently discussed words and phrases")
            keywords = extract_keywords(df["comment"].tolist())
            if keywords:
                st.dataframe(pd.DataFrame(keywords), use_container_width=True, hide_index=True)
            else:
                st.info("Not enough text to extract keywords.")

elif page == "Priority Review":
    st.title("Priority Feedback Review")
    consultation_id = choose_consultation()
    if consultation_id is not None:
        records = get_feedback(consultation_id=consultation_id)
        flagged = [r for r in records if r["is_priority"]]
        st.metric("Comments flagged for review", len(flagged))
        if flagged:
            for item in flagged:
                with st.container(border=True):
                    st.write(f'**{item["section_name"]}** — {item["sentiment"]}')
                    st.write(item["comment"])
                    st.caption(item["priority_reason"])
        else:
            st.success("No comments are currently flagged for review.")

elif page == "Feedback Explorer":
    st.title("Feedback Explorer")
    consultation_id = choose_consultation()
    if consultation_id is not None:
        sections = get_sections(consultation_id)
        options = ["All sections"] + [s["section_name"] for s in sections]
        chosen = st.selectbox("Filter by section", options)
        section_id = None
        if chosen != "All sections":
            section_id = get_section_id(consultation_id, chosen)
        records = get_feedback(consultation_id=consultation_id, section_id=section_id)
        df = pd.DataFrame(records)
        if not df.empty:
            sentiment_options = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
            sentiment_filter = st.selectbox("Filter by sentiment", sentiment_options)
            if sentiment_filter != "All":
                df = df[df["sentiment"] == sentiment_filter]
            search = st.text_input("Search comment text")
            if search:
                df = df[df["comment"].str.contains(search, case=False, na=False)]
            st.dataframe(
                df[["created_at", "section_name", "comment", "sentiment", "is_priority", "priority_reason"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No matching feedback found.")
