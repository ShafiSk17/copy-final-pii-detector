"""
streamlit_app.py

Streamlit UI for manual review + dynamic regex pattern additions for the
DataPipeline (loader -> classifier -> anonymizer).

Usage:
    streamlit run streamlit_app.py

Assumes: data_pipeline.py is in same directory and defines DataPipeline and default_regex_patterns.
"""
import re
import streamlit as st
import tempfile
import json
import os
from pathlib import Path
from typing import List

# Import your pipeline (ensure data_pipeline.py is accessible)
from data_pipeline import DataPipeline, default_regex_patterns

st.set_page_config(page_title="PII Data Pipeline Demo", layout="wide")

# --------------------------
# Helpers
# --------------------------
@st.cache_resource
def get_pipeline(output_dir: str = None, chunk_size: int = 1500, chunk_overlap: int = 200):
    p = DataPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap, output_dir=output_dir)
    # Register default detectors and anonymizers
    p.register_regex_patterns(default_regex_patterns.copy())
    p.register_spacy("en_core_web_sm")
    p.register_presidio()
    p.register_ml()  # placeholder
    return p

def save_uploaded_file(uploaded_file, dst_folder):
    dst = Path(dst_folder) / uploaded_file.name
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dst)

def make_download_link(text: str, filename: str):
    b = text.encode('utf-8')
    return st.download_button(label=f"Download {filename}", data=b, file_name=filename, mime="text/plain")

# --------------------------
# Sidebar: settings + pattern editor
# --------------------------
st.sidebar.header("Pipeline Settings")
out_dir = st.sidebar.text_input("Output dir (local)", value="./st_pipeline_out")
chunk_size = st.sidebar.number_input("Chunk size", value=1500, min_value=500, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", value=200, min_value=0, step=50)
ensure = Path(out_dir)
ensure.mkdir(parents=True, exist_ok=True)

st.sidebar.markdown("---")
st.sidebar.header("Regex Patterns (live)")

# Keep patterns in session_state
if "regex_patterns" not in st.session_state:
    st.session_state.regex_patterns = default_regex_patterns.copy()

# Show current patterns
with st.sidebar.expander("Current Patterns", expanded=True):
    for k, v in st.session_state.regex_patterns.items():
        st.text_input(f"Pattern: {k}", value=v, key=f"pattern_{k}")

# Add new pattern
st.sidebar.markdown("Add new regex pattern")
new_label = st.sidebar.text_input("Label (e.g., SSN)")
new_pattern = st.sidebar.text_input("Pattern (Python regex)")
if st.sidebar.button("Add pattern"):
    if new_label and new_pattern:
        st.session_state.regex_patterns[new_label] = new_pattern
        st.success(f"Added pattern: {new_label}")
    else:
        st.sidebar.error("Provide both label and pattern.")

if st.sidebar.button("Reset to defaults"):
    st.session_state.regex_patterns = default_regex_patterns.copy()
    st.sidebar.success("Reset to default patterns (session only).")

st.sidebar.markdown("---")
st.sidebar.write("Click **Run pipeline** on the main page after uploading files.")

# --------------------------
# Main UI
# --------------------------
st.title("PII Detection & Anonymization — Demo UI")
st.markdown(
    """
    Upload files or provide local file paths. The pipeline will:
    1. Load files lazily (PDF/CSV/JSON/IMG/TXT/EML)
    2. Chunk text
    3. Run Regex / spaCy / Presidio / ML detectors
    4. Produce anonymized outputs (per anonymizer)
    """
)

uploaded = st.file_uploader("Upload files (multiple)", accept_multiple_files=True)

# Optionally allow user to run on server-local paths (for large files)
st.write("Or list local file paths (one per line):")
local_paths_input = st.text_area("Local file paths (optional)", height=80, placeholder="./data/file1.pdf\n./data/file2.csv")

# Button to initialize pipeline with chosen settings
if st.button("Initialize/Update Pipeline"):
    # Recreate pipeline with new output dir/chunk settings
    st.session_state.pipeline = get_pipeline(output_dir=out_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Update patterns in pipeline
    # Remove previous regex classifier/anonymizer and re-register (simple approach: re-init pipeline)
    # For simplicity, we will re-register regex patterns by creating a new pipeline instance
    # NOTE: get_pipeline caches per args; to change patterns we manually register
    p = st.session_state.pipeline
    # Remove previously registered regex ones by re-creating pipeline would be ideal; here, just register new patterns
    p.register_regex_patterns(st.session_state.regex_patterns.copy())
    st.success("Pipeline initialized/updated. Patterns registered.")

# lazy init if not already
if "pipeline" not in st.session_state:
    st.session_state.pipeline = get_pipeline(output_dir=out_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # ensure patterns loaded
    st.session_state.pipeline.register_regex_patterns(st.session_state.regex_patterns.copy())

# Prepare sources list
tmpdir = tempfile.mkdtemp(prefix="st_pipeline_")
sources: List = []
if uploaded:
    st.write(f"Uploaded {len(uploaded)} file(s). Saving temporarily...")
    for f in uploaded:
        saved = save_uploaded_file(f, tmpdir)
        sources.append(saved)

# include local paths
if local_paths_input.strip():
    lines = [l.strip() for l in local_paths_input.splitlines() if l.strip()]
    for p in lines:
        sources.append(p)

st.write("Files to process:")
if not sources:
    st.info("No files selected yet. Upload files or add local paths, then click Run pipeline.")
else:
    for s in sources:
        st.write(f"- {s}")

# Run the pipeline
if st.button("Run pipeline"):
    if not sources:
        st.error("No files provided.")
    else:
        pipeline: DataPipeline = st.session_state.pipeline
        # update regex patterns before run (ensure latest)
        pipeline.register_regex_patterns(st.session_state.regex_patterns.copy())

        st.info("Running pipeline — this may take a while for large files.")
        # Run and show a progress bar
        progress = st.progress(0)
        results = []
        try:
            total = len(sources)
            # run per-source to show partial progress
            for i, src in enumerate(sources):
                st.write(f"Processing: {src}")
                res = pipeline.run_batch([src], save_outputs=True)
                results.extend(res)
                progress.progress(int((i + 1) / total * 100))
            st.success("Pipeline run complete.")
        except Exception as e:
            st.exception(e)
            st.error("Pipeline failed. See logs for details.")

        # Show results summary
        st.header("Run Results Summary")
        if not results:
            st.warning("No metadata returned.")
        else:
            for r in results:
                with st.expander(f"Source: {r.get('source')} — Dominant: {r.get('dominant_type')}"):
                    st.write("Document type:", r.get("doc_type"))
                    st.write("Processed at:", r.get("processed_at"))
                    st.write("Num chunks:", r.get("num_chunks"))
                    st.write("Entity counts:", r.get("entity_counts"))
                    # show chunk summaries
                    st.write("Chunks (first 5):")
                    for c in r.get("chunks", [])[:5]:
                        st.markdown("----")
                        st.write("Chunk metadata:", c.get("chunk_meta"))
                        st.write("Text snippet:", c.get("text_snippet"))
                        st.write("Classifications:")
                        for clf in c.get("classifications", []):
                            st.write(" -", clf.get("classifier"))
                            # show up to 10 entities per classifier
                            for ent in clf.get("results", [])[:10]:
                                st.write("    •", ent.get("type"), "=>", (ent.get("value")[:120] + "...") if len(str(ent.get("value"))) > 120 else ent.get("value"))

                    st.markdown("### Anonymized versions (preview)")
                    for name, anon_preview in r.get("anonymized_versions", {}).items():
                        st.write(f"**{name}** (preview)")
                        st.code(anon_preview[:2000])
                        # full download button (load actual file from output_dir)
                        base = Path(r.get("source")).name
                        safe_base = re.sub(r"[^\w\-_.]", "_", base)
                        file_path = Path(out_dir) / f"{safe_base}.{name}.anonymized.txt"
                        if file_path.exists():
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            st.download_button(f"Download {name} anonymized", data=content, file_name=f"{safe_base}.{name}.anonymized.txt")
                        else:
                            st.write("No file saved for this anonymizer.")

# Footer
st.markdown("---")
st.write("Notes:")
st.write("- Patterns you add are stored in your session only (not persisted to disk).")
st.write("- For production, persist pattern store (DB) and allow versioning/audit of pattern changes.")
