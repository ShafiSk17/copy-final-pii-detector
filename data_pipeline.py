"""
data_pipeline.py

End-to-end streaming data loader + classifier + anonymizer pipeline.
- Lazy loading for many file types (PDF, CSV, JSON, XML, TXT/LOG, images via OCR, SQL tables, emails)
- Chunking via LangChain RecursiveCharacterTextSplitter
- Multi-detector classification: Regex, spaCy NER, Presidio, ML placeholder
- Multi-anonymizer: Presidio, Regex, spaCy, ML placeholder
- Aggregates metadata, optional saving of anonymized files and metadata JSON

Usage:
    from data_pipeline import DataPipeline, default_regex_patterns
    pipeline = DataPipeline(output_dir="./out")
    pipeline.register_regex_patterns(default_regex_patterns)
    pipeline.run_batch(["/path/to/file.pdf", {"type":"sql","conn":"sqlite:///./db","table":"users"}])
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Generator, Union
from datetime import datetime
from pathlib import Path

# File / text loaders
import magic
import pandas as pd
from pypdf import PdfReader
from PIL import Image
import pytesseract
import mailparser
from sqlalchemy import create_engine, text

# LangChain Document + splitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# NLP / Detection libraries
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-pipeline")

# ---------- Defaults (change as needed) ----------
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

default_regex_patterns = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE_10": r"\b\d{10}\b",
    # PAN (India) approx pattern (simple)
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    # Aadhaar (India) - simple 12-digit pattern
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    # Credit card (very simplified)
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
}

# ---------- Utilities ----------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

# ---------- Document factory ----------
def make_doc(text: str, metadata: Dict[str, Any]):
    return Document(page_content=text, metadata=metadata)

# ---------- Text Splitter ----------
def get_text_splitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

# ---------- Lazy Loaders (file / DB) ----------
def detect_mime(path: str) -> str:
    try:
        return magic.from_file(path, mime=True)
    except Exception:
        return ""

def load_pdf(path: str) -> Generator[Document, None, None]:
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        yield make_doc(text, {"source": path, "page": i, "type": "pdf"})

def load_csv(path: str) -> Generator[Document, None, None]:
    # read in chunks if big file
    for chunk in pd.read_csv(path, dtype=str, chunksize=1000):
        for idx, row in chunk.iterrows():
            text = ", ".join([f"{c}: {row[c]}" for c in chunk.columns])
            yield make_doc(text, {"source": path, "row": int(idx), "type": "csv"})

def load_json(path: str) -> Generator[Document, None, None]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    def recurse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from recurse(v, prefix=f"{prefix}{k}.")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                yield from recurse(item, prefix=f"{prefix}[{i}].")
        else:
            yield f"{prefix.rstrip('.')}: {obj}"

    flat = "\n".join(recurse(payload))
    yield make_doc(flat, {"source": path, "type": "json"})

def load_xml(path: str) -> Generator[Document, None, None]:
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()

    def recurse(elem, prefix=""):
        tag = f"{prefix}{elem.tag}"
        texts = []
        if elem.text and elem.text.strip():
            texts.append(f"{tag}: {elem.text.strip()}")
        for child in elem:
            texts.extend(recurse(child, prefix=f"{tag}."))
        return texts

    flat = "\n".join(recurse(root))
    yield make_doc(flat, {"source": path, "type": "xml"})

def load_log(path: str) -> Generator[Document, None, None]:
    # stream lines to avoid memory spikes
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        buf = []
        buf_lines = 0
        chunk_lines = 500
        for i, line in enumerate(f):
            buf.append(line)
            buf_lines += 1
            if buf_lines >= chunk_lines:
                yield make_doc("".join(buf), {"source": path, "type": "log", "chunk_index": i // chunk_lines})
                buf = []
                buf_lines = 0
        if buf:
            yield make_doc("".join(buf), {"source": path, "type": "log", "chunk_index": (i // chunk_lines)})

def load_image(path: str) -> Generator[Document, None, None]:
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    yield make_doc(text, {"source": path, "type": "image", "ocr": True})

def load_plain_text(path: str) -> Generator[Document, None, None]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        buf = []
        buf_lines = 0
        chunk_lines = 500
        for i, line in enumerate(f):
            buf.append(line)
            buf_lines += 1
            if buf_lines >= chunk_lines:
                yield make_doc("".join(buf), {"source": path, "type": "text", "chunk_index": i // chunk_lines})
                buf = []
                buf_lines = 0
        if buf:
            yield make_doc("".join(buf), {"source": path, "type": "text", "chunk_index": (i // chunk_lines)})

def load_email(path: str) -> Generator[Document, None, None]:
    parsed = mailparser.parse_from_file(path)
    body = parsed.body or ""
    metadata = {
        "source": path,
        "type": "email",
        "from": parsed.from_,
        "to": parsed.to,
        "subject": parsed.subject,
        "date": parsed.date,
    }
    yield make_doc(body, metadata)

def load_sql_table(conn_str: str, table: str) -> Generator[Document, None, None]:
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table}"))
        cols = result.keys()
        for idx, row in enumerate(result):
            text_row = ", ".join([f"{c}: {row[c]}" for c in cols])
            yield make_doc(text_row, {"source": f"{conn_str}::{table}", "row": idx, "type": "sql"})

# Dispatcher
def load_file(path: str) -> Generator[Document, None, None]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        yield from load_pdf(path)
    elif ext == ".csv":
        yield from load_csv(path)
    elif ext == ".json":
        yield from load_json(path)
    elif ext == ".xml":
        yield from load_xml(path)
    elif ext in [".log", ".txt"]:
        mime = detect_mime(path)
        if mime and mime.startswith("text"):
            yield from load_plain_text(path)
        else:
            # fallback to text
            yield from load_plain_text(path)
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        yield from load_image(path)
    elif ext in [".eml", ".msg"]:
        yield from load_email(path)
    else:
        # try mime detection
        mime = detect_mime(path)
        if mime:
            if "pdf" in mime:
                yield from load_pdf(path)
            elif mime.startswith("image"):
                yield from load_image(path)
            elif mime.startswith("text"):
                yield from load_plain_text(path)
            else:
                yield from load_plain_text(path)
        else:
            yield from load_plain_text(path)

def load_batch(sources: List[Union[str, Dict[str, Any]]]) -> Generator[Document, None, None]:
    for src in sources:
        if isinstance(src, dict) and src.get("type") == "sql":
            yield from load_sql_table(src["conn"], src["table"])
        else:
            yield from load_file(src)

# ---------- Classifiers (Regex, SpaCy NER, Presidio, ML placeholder) ----------
class BaseClassifier:
    def classify(self, text: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

class RegexClassifier(BaseClassifier):
    def __init__(self, patterns: Dict[str, str] = None):
        self.patterns = patterns or {}

    def add_pattern(self, label: str, pattern: str):
        self.patterns[label] = pattern

    def classify(self, text: str):
        results = []
        for label, pattern in self.patterns.items():
            try:
                for match in re.finditer(pattern, text):
                    results.append({"type": label, "start": match.start(), "end": match.end(), "value": match.group()})
            except re.error as e:
                logger.warning(f"Invalid regex for {label}: {e}")
        return results

class SpaCyNERClassifier(BaseClassifier):
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def classify(self, text: str):
        doc = self.nlp(text)
        return [{"type": ent.label_, "start": ent.start_char, "end": ent.end_char, "value": ent.text} for ent in doc.ents]

class PresidioClassifier(BaseClassifier):
    def __init__(self):
        self.engine = AnalyzerEngine()

    def classify(self, text: str):
        results = self.engine.analyze(text=text, language="en")
        out = []
        for r in results:
            out.append({"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score, "value": text[r.start:r.end]})
        return out

class MLClassifier(BaseClassifier):
    """
    Placeholder: plug your own token-level model (NER) or document-level model.
    For token-level entity detection, integrate a fine-tuned transformer that returns spans.
    """
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def classify(self, text: str):
        # If not configured, return empty
        return []

class ClassifierManager:
    def __init__(self):
        self.classifiers: List[BaseClassifier] = []

    def register(self, clf: BaseClassifier):
        self.classifiers.append(clf)

    def classify(self, text: str):
        all_results = []
        for clf in self.classifiers:
            try:
                res = clf.classify(text)
                if res:
                    all_results.append({"classifier": clf.__class__.__name__, "results": res})
            except Exception as e:
                logger.exception(f"Classifier {clf.__class__.__name__} failed: {e}")
        return all_results

# ---------- Anonymizers ----------
class BaseAnonymizer:
    def anonymize(self, text: str) -> str:
        raise NotImplementedError

class RegexAnonymizer(BaseAnonymizer):
    def __init__(self, patterns: Dict[str, str]):
        # We'll replace matches with label-based mask
        self.patterns = patterns

    def anonymize(self, text: str) -> str:
        out = text
        for label, pattern in self.patterns.items():
            try:
                out = re.sub(pattern, f"<{label}_MASKED>", out)
            except re.error:
                logger.warning(f"Invalid regex pattern for anonymizer: {label}")
        return out

class SpaCyAnonymizer(BaseAnonymizer):
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def anonymize(self, text: str) -> str:
        doc = self.nlp(text)
        out = text
        # replace in reverse order to keep indexes valid
        for ent in reversed(doc.ents):
            mask = f"<{ent.label_}_MASKED>"
            out = out[:ent.start_char] + mask + out[ent.end_char:]
        return out

class PresidioAnonymizer(BaseAnonymizer):
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def anonymize(self, text: str) -> str:
        analysis_results = self.analyzer.analyze(text=text, language="en")
        if not analysis_results:
            return text
        # build anonymizers mapping: replace with label mask
        mapping = {}
        for r in analysis_results:
            mapping.setdefault(r.entity_type, {"type": "replace", "new_value": f"<{r.entity_type}_MASKED>"})
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=analysis_results, anonymizers=mapping)
        return anonymized.text

class MLModelAnonymizer(BaseAnonymizer):
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def anonymize(self, text: str) -> str:
        # Not implemented by default
        return text

class AnonymizerManager:
    def __init__(self):
        self.anonymizers: List[BaseAnonymizer] = []

    def register(self, anonymizer: BaseAnonymizer):
        self.anonymizers.append(anonymizer)

    def anonymize(self, text: str) -> Dict[str, str]:
        out = {}
        for a in self.anonymizers:
            try:
                out[a.__class__.__name__] = a.anonymize(text)
            except Exception as e:
                logger.exception(f"Anonymizer {a.__class__.__name__} failed: {e}")
                out[a.__class__.__name__] = text
        return out

# ---------- Data Pipeline ----------
class DataPipeline:
    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, output_dir: str = None):
        self.splitter = get_text_splitter(chunk_size, chunk_overlap)
        self.classifier_mgr = ClassifierManager()
        self.anonymizer_mgr = AnonymizerManager()
        self.output_dir = output_dir
        if output_dir:
            ensure_dir(output_dir)

    # helpers to register components
    def register_regex_patterns(self, patterns: Dict[str, str]):
        regex_clf = RegexClassifier(patterns)
        self.classifier_mgr.register(regex_clf)
        self.anonymizer_mgr.register(RegexAnonymizer(patterns))

    def register_spacy(self, model_name="en_core_web_sm"):
        spacy_clf = SpaCyNERClassifier(model_name=model_name)
        self.classifier_mgr.register(spacy_clf)
        self.anonymizer_mgr.register(SpaCyAnonymizer(model_name=model_name))

    def register_presidio(self):
        pres_clf = PresidioClassifier()
        self.classifier_mgr.register(pres_clf)
        self.anonymizer_mgr.register(PresidioAnonymizer())

    def register_ml(self, model=None, tokenizer=None):
        ml_clf = MLClassifier(model=model, tokenizer=tokenizer)
        ml_anon = MLModelAnonymizer(model=model, tokenizer=tokenizer)
        self.classifier_mgr.register(ml_clf)
        self.anonymizer_mgr.register(ml_anon)

    # core pipeline for a single Document (LangChain Document-like)
    def process_document(self, doc: Document, save_outputs: bool = True) -> Dict[str, Any]:
        """
        Process one Document: split -> classify each chunk -> aggregate -> anonymize original text -> save
        Returns metadata dict.
        """
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "unknown")
        logger.info(f"Processing doc from {source} (type={doc_type})")

        # 1) split into chunks
        chunks = self.splitter.split_documents([doc])  # returns list of Documents
        chunk_results = []
        aggregated_entities = []

        for chunk in chunks:
            text = chunk.page_content
            # 2) classify chunk
            classifications = self.classifier_mgr.classify(text)
            chunk_results.append({"chunk_meta": chunk.metadata, "text_snippet": text[:200], "classifications": classifications})

            # aggregate flat list of entities for metadata
            for c in classifications:
                for ent in c["results"]:
                    aggregated_entities.append({"classifier": c["classifier"], **ent})

        # 3) produce anonymized outputs using all registered anonymizers
        original_text = doc.page_content
        anonymized_versions = self.anonymizer_mgr.anonymize(original_text)

        # 4) compute dominant category heuristics simple: count entity types (very basic)
        type_counts = {}
        for ent in aggregated_entities:
            t = ent.get("type") or ent.get("entity_type") or "UNKNOWN"
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None

        # 5) prepare metadata
        metadata = {
            "source": source,
            "doc_type": doc_type,
            "processed_at": now_iso(),
            "num_chunks": len(chunk_results),
            "entity_counts": type_counts,
            "dominant_type": dominant_type,
            "chunks": chunk_results,
            "anonymized_versions": {k: (v[:1000] + "...") if len(v) > 1000 else v for k, v in anonymized_versions.items()}
        }

        # 6) save outputs if requested
        if save_outputs and self.output_dir:
            base_name = Path(source).name
            safe_base = re.sub(r"[^\w\-_.]", "_", base_name)
            # save anonymized versions (each anonymizer)
            for name, anon_text in anonymized_versions.items():
                out_path = Path(self.output_dir) / f"{safe_base}.{name}.anonymized.txt"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(anon_text)
            # save metadata
            meta_path = Path(self.output_dir) / f"{safe_base}.metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        return metadata

    # process a batch of sources
    def run_batch(self, sources: List[Union[str, Dict[str, Any]]], save_outputs: bool = True):
        docs = load_batch(sources)
        results = []
        for doc in docs:
            # each doc is a LangChain Document
            try:
                meta = self.process_document(doc, save_outputs=save_outputs)
                results.append(meta)
            except Exception as e:
                logger.exception(f"Failed to process doc {doc.metadata.get('source')}: {e}")
        return results

# ---------- Example usage ----------
if __name__ == "__main__":
    # Create pipeline
    pipeline = DataPipeline(output_dir="./pipeline_out")

    # Register detection + anonymization methods
    pipeline.register_regex_patterns(default_regex_patterns)
    pipeline.register_spacy("en_core_web_sm")
    pipeline.register_presidio()
    pipeline.register_ml()  # optional; placeholder

    # Sources: files and an example SQL source dict
    sources = [
        "./examples/sample.pdf",
        "./examples/sample.csv",
        "./examples/sample.png",
        {"type": "sql", "conn": "sqlite:///./examples/example.db", "table": "users"},
        "./examples/sample.eml"
    ]

    logger.info("Starting batch run...")
    out = pipeline.run_batch(sources, save_outputs=True)
    logger.info("Run complete. Metadata summary:")
    print(json.dumps(out, indent=2))
