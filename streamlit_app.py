"""
streamlit_app.py

Streamlit UI for PII pipeline:
- Clean landing page
- Choose file-type to upload (PDF/CSV/JSON/IMG/TXT/EML)
- Upload files
- Run pipeline per-file
- Table where each classifier is a row and shows detected entities (and optional confidence)
- Preview first chunk with highlighted detections (basic)
- Add new Regex and Presidio patterns via UI; they are applied immediately

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import tempfile
import json
import os
import re
from pathlib import Path
from typing import List, Dict
from data_pipeline import DataPipeline, default_regex_patterns

st.set_page_config(page_title="PII Pipeline — Review UI", layout="wide")
st.title("PII Detection & Anonymization — Manual Review")

# -------------------------
# Session and pipeline init
# -------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = DataPipeline(output_dir="./st_pipeline_out")
    # register default detectors
    st.session_state.pipeline.register_regex_patterns(default_regex_patterns.copy())
    st.session_state.pipeline.register_spacy("en_core_web_sm")
    st.session_state.pipeline.register_presidio()
    st.session_state.pipeline.register_ml()

# Keep patterns persistent in session only
if "regex_patterns" not in st.session_state:
    st.session_state.regex_patterns = default_regex_patterns.copy()

# Presidio custom patterns stored in session (name->regex)
if "presidio_patterns" not in st.session_state:
    st.session_state.presidio_patterns = {}

# Create output dir
Path(st.session_state.pipeline.output_dir or "./st_pipeline_out").mkdir(parents=True, exist_ok=True)

# -------------------------
# Left: controls
# -------------------------
with st.sidebar:
    st.header("Upload & Settings")
    st.markdown("Choose file type and upload matching files. You can upload multiple files of the same type.")

    file_type = st.selectbox("File type", ["pdf", "csv", "json", "image", "text", "eml"])
    uploaded_files = st.file_uploader(f"Upload {file_type} file(s)", accept_multiple_files=True,
                                      type=("pdf" if file_type=="pdf" else
                                            "csv" if file_type=="csv" else
                                            "json" if file_type=="json" else
                                            ("png","jpg","jpeg","tif","tiff") if file_type=="image" else
                                            "txt" if file_type=="text" else
                                            "eml" if file_type=="eml" else None))

    st.markdown("---")
    st.header("Chunk settings")
    chunk_size = st.number_input("Chunk size (characters)", value=1500, min_value=500, step=100)
    chunk_overlap = st.number_input("Chunk overlap (characters)", value=200, min_value=0, step=50)

    if st.button("Update pipeline chunk settings"):
        # re-init pipeline with updated splitter while keeping registered detectors
        # Easiest approach: create new instance, re-register components and patterns
        p = DataPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap, output_dir=st.session_state.pipeline.output_dir)
        # re-register: regex, spacy, presidio, ml
        p.register_regex_patterns(st.session_state.regex_patterns.copy())
        p.register_spacy("en_core_web_sm")
        p.register_presidio()
        p.register_ml()
        # re-add presidio custom patterns if any
        for name, regex in st.session_state.presidio_patterns.items():
            p.add_presidio_pattern(name, regex)
        st.session_state.pipeline = p
        st.success("Pipeline updated with new chunk settings.")

    st.markdown("---")
    st.header("Add Regex Pattern (live)")
    new_label = st.text_input("Pattern label (e.g., SSN)")
    new_regex = st.text_input("Regex (Python)")
    if st.button("Add Regex Pattern"):
        if new_label and new_regex:
            st.session_state.regex_patterns[new_label] = new_regex
            # add to pipeline immediately
            st.session_state.pipeline.add_regex_pattern(new_label, new_regex)
            st.success(f"Added regex: {new_label}")
        else:
            st.error("Provide label and regex.")

    st.markdown("---")
    st.header("Add Presidio Pattern (live)")
    pres_label = st.text_input("Presidio label (e.g., CUSTOM_ID)", key="pres_label")
    pres_regex = st.text_input("Presidio regex (pattern)", key="pres_regex")
    if st.button("Add Presidio Pattern"):
        if pres_label and pres_regex:
            added = st.session_state.pipeline.add_presidio_pattern(pres_label, pres_regex)
            if added:
                st.session_state.presidio_patterns[pres_label] = pres_regex
                st.success(f"Added Presidio pattern: {pres_label}")
            else:
                st.error("Failed to add Presidio pattern. See logs.")
        else:
            st.error("Enter both label and regex.")

    st.markdown("---")
    st.markdown("Patterns stored in session. For persistence, save to DB or file.")

# -------------------------
# Right / Main: file list and run
# -------------------------
st.header("1. Files to process")
tmpdir = tempfile.mkdtemp(prefix="st_pipeline_")
sources: List[str] = []

if uploaded_files:
    for f in uploaded_files:
        save_path = Path(tmpdir) / f.name
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
        sources.append(str(save_path))

st.write("Selected files:")
if not sources:
    st.info("No files uploaded. Select a file type and upload files.")
else:
    for s in sources:
        st.write("-", s)

run_button = st.button("Run pipeline on selected files")

if run_button:
    if not sources:
        st.error("Upload at least one file.")
    else:
        # Ensure pipeline has current regex patterns before run
        st.session_state.pipeline.register_regex_patterns(st.session_state.regex_patterns.copy())
        # Also ensure presidio custom patterns are added to engine (they might already be)
        for name, regex in st.session_state.presidio_patterns.items():
            st.session_state.pipeline.add_presidio_pattern(name, regex)

        st.info("Processing files... (this runs the pipeline inline; for heavy loads use Airflow workers)")
        results = st.session_state.pipeline.run_batch(sources, save_outputs=True)

        if not results:
            st.warning("No results returned (check logs).")
        else:
            # For simplicity show results for the first processed file
            first = results[0]

            st.subheader("2. Classifier results table (aggregated per classifier)")
            # Build a table where each row is a classifier, and detected entities are shown
            classifier_rows = []
            # Use chunk-level classifications in first['chunks']
            # aggregate per classifier name
            agg = {}
            for c in first.get("chunks", []):
                for clf in c.get("classifications", []):
                    name = clf.get("classifier")
                    ents = [r.get("value") for r in clf.get("results", [])]
                    scores = [r.get("score") for r in clf.get("results", []) if r.get("score") is not None]
                    if name not in agg:
                        agg[name] = {"detected": [], "confidences": []}
                    agg[name]["detected"].extend(ents)
                    agg[name]["confidences"].extend([s for s in scores if s is not None])

            # create display rows
            rows = []
            for name, v in agg.items():
                uniq = list(dict.fromkeys([str(x) for x in v["detected"] if x]))
                avg_conf = round(sum(v["confidences"])/len(v["confidences"]), 2) if v["confidences"] else None
                rows.append({"Classifier": name, "Detected (sample)": ", ".join(uniq[:10]), "Confidence (avg)": avg_conf})

            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df)

            st.markdown("---")
            st.subheader("3. Preview — first chunk (with classifier highlights)")

            # find the first non-empty chunk
            preview_chunk = None
            for c in first.get("chunks", []):
                if c.get("text_snippet"):
                    preview_chunk = c
                    break

            if not preview_chunk:
                st.warning("No chunk text available to preview.")
            else:
                text_snip = preview_chunk.get("text_snippet", "")
                st.markdown("**Chunk metadata**")
                st.write(preview_chunk.get("chunk_meta"))

                st.markdown("**Raw snippet**")
                st.code(text_snip[:2000])

                # Build highlighted snippet using simple spans (not true HTML highlighting in Streamlit)
                # We'll produce an HTML with <mark> tags and show via st.markdown(unsafe_allow_html=True)
                highlights = []
                # collect all entity spans from classifiers for this chunk
                spans = []
                for clf in preview_chunk.get("classifications", []):
                    cname = clf.get("classifier")
                    for ent in clf.get("results", []):
                        try:
                            s = int(ent.get("start")) if ent.get("start") is not None else None
                            e = int(ent.get("end")) if ent.get("end") is not None else None
                        except Exception:
                            s = None
                            e = None
                        val = ent.get("value") or ""
                        spans.append({"start": s, "end": e, "label": ent.get("type") or cname, "value": val, "source": cname})

                # sort spans by start position
                spans = [sp for sp in spans if sp["start"] is not None and sp["end"] is not None and sp["end"] > sp["start"]]
                spans = sorted(spans, key=lambda x: x["start"])

                # build html
                def escape_html(s):
                    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                if spans:
                    html = ""
                    cursor = 0
                    for sp in spans:
                        if sp["start"] > cursor:
                            html += escape_html(text_snip[cursor:sp["start"]])
                        # mark label inside <mark> with tooltip via title
                        label = sp["label"]
                        excerpt = escape_html(text_snip[sp["start"]:sp["end"]])
                        html += f"<mark title='{escape_html(sp['source'])} | {escape_html(label)}'>{excerpt} <small style='color:gray'>[{escape_html(label)}]</small></mark>"
                        cursor = sp["end"]
                    if cursor < len(text_snip):
                        html += escape_html(text_snip[cursor:])
                else:
                    html = escape_html(text_snip)

                st.markdown(html, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("4. Anonymized previews (per anonymizer)")
            for name, preview in first.get("anonymized_versions", {}).items():
                st.markdown(f"**{name}**")
                st.code(preview[:2000])
                base = Path(first.get("source")).name
                safe_base = re.sub(r"[^\w\-_.]", "_", base)
                file_path = Path(st.session_state.pipeline.output_dir) / f"{safe_base}.{name}.anonymized.txt"
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    st.download_button(f"Download {name} anonymized", data=content, file_name=f"{safe_base}.{name}.anonymized.txt")

st.markdown("---")
st.write("Notes:")
st.write("- Presidio and Regex patterns added here apply immediately for new runs.")
st.write("- spaCy & ML models must be trained offline and loaded into the pipeline for inference.")

