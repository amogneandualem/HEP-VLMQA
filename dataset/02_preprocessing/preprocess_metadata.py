#!/usr/bin/env python3
"""
HEP-VLMQA metadata preprocessing.

This script converts per-paper `image_caption.jsonl` files into
`clean_image_caption.jsonl` while preserving the original preprocessing logic:

- normalize captions;
- clean surrounding paper context;
- repair common extraction noise;
- preserve short inline formulas and replace long display equations;
- remove duplicated caption text from context;
- remove missing-image records;
- remove captions shorter than three words;
- deduplicate by image path + normalized caption;
- assign rule-based scientific domain labels;
- assign rule-based image-type labels;
- derive similarity groups;
- extract physics keywords.

Expected input
--------------
<root>/<paper_id>/image_caption.jsonl

Expected output
---------------
<root>/<paper_id>/clean_image_caption.jsonl

Example
-------
python preprocess_metadata.py \
    --root /path/to/per_paper_dataset

The scientific classification rules in this release are retained from the
original preprocessing implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("hep_vlmqa.preprocessing")

DEFAULT_INPUT_FILE = "image_caption.jsonl"
DEFAULT_OUTPUT_FILE = "clean_image_caption.jsonl"
DEFAULT_MAX_CONTEXT_CHARS = 5000


DOMAIN_KEYWORDS = {
    "particle_physics": [
        "higgs",
        "decay",
        "sm",
        "hsm",
        "thdm",
        "two higgs doublet",
        "radiative corrections",
        "quantum corrections",
        "higgs boson",
        "branching fraction",
        "cross section",
        "invariant mass",
        "qcd",
        "quark",
        "lepton",
        "boson",
        "lhc",
        "cms",
        "atlas",
    ],
    "astrophysics": [
        "quasar",
        "qso",
        "black hole",
        "m_bh",
        "fe ii",
        "starlight",
        "hβ",
        "galaxy",
        "x-ray",
        "pulsar",
        "supernova",
    ],
    "cosmology": [
        "dark matter",
        "cmb",
        "planck",
        "matter power spectrum",
        "lambdacdm",
        "redshift",
        "inflation",
    ],
    "theoretical_physics": [
        "path integral",
        "density matrix",
        "replica",
        "entropy",
        "lagrangian",
        "hamiltonian",
        "field theory",
    ],
}


IMAGE_TYPE_RULES = {
    "correlation_plot": [
        "correlation",
        "ratios",
        "between",
        "higgs decay widths",
    ],
    "line_plot": [
        "line",
        "tree-level",
        "quantum corrections",
        "radiative corrections",
    ],
    "fit_plot": [
        "fit",
        "best fit",
        "likelihood",
        "residuals",
    ],
    "mass_spectrum": [
        "invariant mass",
        "mass spectrum",
        "mass distribution",
    ],
    "multi_panel_plot": [
        "top panel",
        "bottom panel",
        "left panel",
        "right panel",
        "(a)",
        "(b)",
    ],
    "histogram": [
        "histogram",
        "distribution",
    ],
}


STOPWORDS = set(
    """
the a an and or of to in on for with as by from is are was were be this that
these those it its we our their they which when where while using used case
""".split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean HEP-VLMQA figure-caption metadata and create "
            "clean_image_caption.jsonl files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing per-paper folders.",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSONL filename (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSONL filename (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Maximum cleaned context length in characters "
            f"(default: {DEFAULT_MAX_CONTEXT_CHARS})."
        ),
    )
    parser.add_argument(
        "--keep-missing-images",
        action="store_true",
        help=(
            "Keep records whose referenced image is missing. "
            "By default such records are removed."
        ),
    )
    parser.add_argument(
        "--paper",
        default=None,
        help="Process only one paper folder by name.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index into sorted paper folders.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end index into sorted paper folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--summary-name",
        default="preprocessing_summary.json",
        help="Run-level summary filename written under --root.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: Any) -> int:
    return len(re.findall(r"\b\w+\b", str(text or "")))


def words(text: Any) -> list[str]:
    return [
        word
        for word in re.findall(
            r"[A-Za-z0-9α-ωΑ-Ω_]+",
            str(text or "").lower(),
        )
        if word not in STOPWORDS
    ]


def overlap_ratio(a: str, b: str) -> float:
    a_words = set(words(a))
    b_words = set(words(b))

    if not a_words or not b_words:
        return 0.0

    return len(a_words & b_words) / max(1, len(a_words))


def protect_known_math(text: str) -> str:
    replacements = [
        (r"h\s*Z\s*Z\s*\^\s*\*", "h → ZZ*"),
        (r"h\s*f\s*f", "h → ff"),
        (r"h\s*Z\s*Z", "h → ZZ"),
        (r"R\s*_\s*Z\s*Z\s*\^\s*\*", "R_ZZ*"),
        (r"R\s*z\s*z\s*:\s*\*", "R_ZZ*"),
        (r"R\s*_\s*f\s*f", "R_ff"),
        (r"m\s*_\s*H", "m_H"),
        (r"s\s*_\s*-", "s_-"),
        (r"c\s*_\s*-", "c_-"),
        (r"tan\s*=", "tanβ ="),
        (r"T\s*H\s*D\s*M", "THDM"),
        (r"H\s*S\s*M", "HSM"),
        (r"S\s*M", "SM"),
        (r"G\s*e\s*V", "GeV"),
        (r"T\s*e\s*V", "TeV"),
        (r"\\Delta\s*\\chi\s*\^\s*\{\s*2\s*\}", "Δχ2"),
        (r"\\sqrt\s*\{\s*s\s*\}", "√s"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text


def clean_latex_vars(text: str) -> str:
    text = protect_known_math(text)

    # Preserve short inline formulas as readable text instead of deleting.
    text = re.sub(
        r"\$([^$]{1,120})\$",
        r" \1 ",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\(([^)]{1,120})\\\)",
        r" \1 ",
        text,
        flags=re.DOTALL,
    )

    # Drop long display equations.
    text = re.sub(
        r"\$\$.*?\$\$",
        " [FORMULA] ",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\[.*?\\\]",
        " [FORMULA] ",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\begin\{equation\}.*?\\end\{equation\}",
        " [FORMULA] ",
        text,
        flags=re.DOTALL,
    )

    # Remove LaTeX commands while retaining their textual content.
    text = re.sub(
        r"\\mathrm\s*\{\s*([^{}]+)\s*\}",
        r"\1",
        text,
    )
    text = re.sub(
        r"\\mathbf\s*\{\s*([^{}]+)\s*\}",
        r"\1",
        text,
    )
    text = re.sub(
        r"\\mathtt\s*\{\s*([^{}]+)\s*\}",
        r"\1",
        text,
    )
    text = re.sub(r"\\big\s*;?", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ")

    return protect_known_math(text)


def repair_common_noise(text: str) -> str:
    fixes = {
        "similiar": "similar",
        "for the our": "for our",
        "statisti-cal": "statistical",
        "hiddenstrangeness": "hidden-strangeness",
        "thte": "the",
        "up-todate": "up-to-date",
        "i. e": "i.e.",
        "7.0.0 ~ GeV": "700 GeV",
        "1 \\ TeV": "1 TeV",
        "1.0 \\%": "1.0%",
        "1.5 \\%": "1.5%",
        "6.0 \\%": "6.0%",
        "m _ H": "m_H",
        "s _ -": "s_-",
        "c _ -": "c_-",
        "R _ f f": "R_ff",
        "R _ Z Z ^ *": "R_ZZ*",
        "h Z Z ^ *": "h → ZZ*",
    }

    for source, replacement in fixes.items():
        text = text.replace(source, replacement)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z])", ". ", text)
    text = re.sub(r"\.{3,}", "...", text)

    return norm(text)


def clean_text(text: Any) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\n", " ").replace("\t", " ")

    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    text = clean_latex_vars(text)
    text = re.sub(r"\[[0-9,\-\s]+\]", " ", text)
    text = repair_common_noise(text)

    text = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", text)
    return norm(text)


def normalize_caption(caption: Any) -> str:
    caption = clean_text(caption)
    caption = re.sub(
        r"^Figure\s+",
        "FIG. ",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"^Fig\.\s*",
        "FIG. ",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"^FIGURE\s+",
        "FIG. ",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"^(FIG\.\s*\d+[a-zA-Z]?)\s*[:\-–—]\s*",
        r"\1. ",
        caption,
        flags=re.IGNORECASE,
    )
    return norm(caption)


def split_sentences(text: Any) -> list[str]:
    text = norm(text)

    protected = {
        "FIG.": "FIG<DOT>",
        "Fig.": "Fig<DOT>",
        "Eq.": "Eq<DOT>",
        "Sec.": "Sec<DOT>",
        "Ref.": "Ref<DOT>",
        "Refs.": "Refs<DOT>",
        "i.e.": "i<DOT>e<DOT>",
        "e.g.": "e<DOT>g<DOT>",
        "vs.": "vs<DOT>",
    }

    for source, replacement in protected.items():
        text = text.replace(source, replacement)

    parts = re.split(r"(?<=[.!?])\s+", text)
    output: list[str] = []

    for sentence in parts:
        for source, replacement in protected.items():
            sentence = sentence.replace(replacement, source)

        sentence = norm(sentence)
        if word_count(sentence) >= 3:
            output.append(sentence)

    return output


def remove_headers_keep_text(text: str) -> str:
    text = re.sub(
        r"#{1,6}\s*\d*\.?\s*[A-Za-z0-9]"
        r"[A-Za-z0-9\s:\-–—]{0,80}",
        " ",
        text,
    )
    return norm(text)


def cut_bad_tail(text: str) -> str:
    bad_patterns = [
        r"\backnowledg(?:e)?ments?\b",
        r"\breferences\b",
        r"\bbibliography\b",
        r"\bfunding\b",
        r"\bsupported by\b",
        r"\bwe thank\b",
        r"\ball rights reserved\b",
        r"\bcopyright\b",
        r"\bthis conference has been organized\b",
        r"\bthe authors would like to acknowledge\b",
        r"\breceived funding from\b",
        r"\bgrant agreement\b",
    ]

    lowered = text.lower()
    cuts = [
        match.start()
        for pattern in bad_patterns
        if (match := re.search(pattern, lowered))
    ]

    return norm(text[: min(cuts)]) if cuts else norm(text)


def remove_caption_prefix(caption: str, description: str) -> str:
    if not description:
        return ""

    caption = norm(caption)
    description = norm(description)

    if not caption:
        return description

    if description.lower().startswith(caption.lower()):
        return norm(description[len(caption) :])

    sentences = split_sentences(description)
    best_cut = 0

    for count in range(1, min(8, len(sentences)) + 1):
        prefix = " ".join(sentences[:count])
        if overlap_ratio(caption, prefix) > 0.70:
            best_cut = count

    if best_cut:
        return norm(" ".join(sentences[best_cut:]))

    return description


def clean_context(
    raw_description: Any,
    caption: str,
    raw_caption: Any,
    max_context_chars: int,
) -> str:
    description = clean_text(raw_description)
    raw_caption_clean = normalize_caption(raw_caption)

    description = remove_headers_keep_text(description)
    description = cut_bad_tail(description)

    description = remove_caption_prefix(caption, description)
    description = remove_caption_prefix(
        raw_caption_clean,
        description,
    )

    cleaned: list[str] = []

    for sentence in split_sentences(description):
        if word_count(sentence) < 3:
            continue

        if re.fullmatch(
            r"[A-Z0-9\s:\-–—]{15,}",
            sentence,
        ):
            continue

        cleaned.append(sentence)

    context = norm(" ".join(cleaned))

    if len(context) > max_context_chars:
        context = context[:max_context_chars].rsplit(" ", 1)[0]

    return norm(context)


def classify(text: str, rules: dict[str, list[str]]) -> str:
    lowered = text.lower()
    scores: Counter[str] = Counter()

    for label, keywords in rules.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[label] += 1

    return scores.most_common(1)[0][0] if scores else "other"


def classify_domain(text: str) -> str:
    lowered = text.lower()

    if (
        sum(
            item in lowered
            for item in (
                "higgs",
                "thdm",
                "hsm",
                "sm-like",
                "decay width",
                "zz*",
                "h → zz",
            )
        )
        >= 2
    ):
        return "particle_physics"

    if (
        sum(
            item in lowered
            for item in (
                "quasar",
                "qso",
                "m_bh",
                "black hole",
                "fe ii",
                "hβ",
            )
        )
        >= 2
    ):
        return "astrophysics"

    if (
        sum(
            item in lowered
            for item in (
                "dark matter",
                "cmb",
                "planck",
                "matter power spectrum",
            )
        )
        >= 2
    ):
        return "cosmology"

    return classify(text, DOMAIN_KEYWORDS)


def classify_image_type(text: str) -> str:
    lowered = text.lower()

    if (
        "correlation" in lowered
        and "higgs decay widths" in lowered
    ):
        return "correlation_plot"

    if "m_bh" in lowered or "black hole mass" in lowered:
        return "black_hole_mass_distribution"

    if "hβ" in lowered or "fe ii template" in lowered:
        return "spectral_fit_plot"

    return classify(text, IMAGE_TYPE_RULES)


def extract_keywords(text: str) -> list[str]:
    lowered = text.lower()
    found: set[str] = set()

    for keywords in DOMAIN_KEYWORDS.values():
        for keyword in keywords:
            if keyword in lowered:
                found.add(keyword)

    return sorted(found)


def similarity_group(image_type: str) -> str:
    return {
        "correlation_plot": "correlation",
        "line_plot": "line_plot",
        "fit_plot": "fit_plot",
        "mass_spectrum": "mass_spectrum",
        "multi_panel_plot": "multi_panel",
        "histogram": "distribution",
        "black_hole_mass_distribution": "distribution",
        "spectral_fit_plot": "fit_plot",
    }.get(image_type, "general")


def process_paper(
    folder: Path,
    input_name: str,
    output_name: str,
    max_context_chars: int,
    drop_missing_images: bool,
    overwrite: bool,
) -> dict[str, Any]:
    source = folder / input_name

    if not source.is_file():
        return {
            "paper_id": folder.name,
            "status": "missing_input",
            "kept": 0,
            "removed": 0,
        }

    output_path = folder / output_name

    if output_path.exists() and not overwrite:
        return {
            "paper_id": folder.name,
            "status": "skipped_existing",
            "kept": 0,
            "removed": 0,
        }

    temporary = output_path.with_name(output_path.name + ".tmp")
    seen: set[str] = set()

    kept = 0
    removed = 0
    invalid_json = 0
    missing_image = 0
    short_caption = 0
    duplicate = 0

    with source.open("r", encoding="utf-8") as source_handle, \
            temporary.open("w", encoding="utf-8") as output_handle:

        for line_number, line in enumerate(
            source_handle,
            start=1,
        ):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                invalid_json += 1
                LOGGER.debug(
                    "Invalid JSON: %s:%d",
                    source,
                    line_number,
                )
                continue

            if not isinstance(item, dict):
                removed += 1
                invalid_json += 1
                continue

            paper_id = item.get("paper_id", folder.name)
            image_relative = str(
                item.get("image", "")
            ).strip()

            if not image_relative:
                removed += 1
                missing_image += 1
                continue

            if (
                drop_missing_images
                and not (folder / image_relative).is_file()
            ):
                removed += 1
                missing_image += 1
                continue

            raw_caption = str(item.get("caption", ""))
            raw_description = str(
                item.get("description", "")
            )

            caption = normalize_caption(raw_caption)

            if word_count(caption) < 3:
                removed += 1
                short_caption += 1
                continue

            key = norm(
                (image_relative + "|" + caption).lower()
            )

            if key in seen:
                removed += 1
                duplicate += 1
                continue

            seen.add(key)

            context = clean_context(
                raw_description=raw_description,
                caption=caption,
                raw_caption=raw_caption,
                max_context_chars=max_context_chars,
            )

            combined = caption + " " + context
            image_type = classify_image_type(combined)
            domain = classify_domain(combined)

            record = {
                "paper_id": paper_id,
                "image": image_relative,
                "domain": domain,
                "image_type": image_type,
                "similarity_group": similarity_group(
                    image_type
                ),
                "physics_keywords": extract_keywords(
                    combined
                ),
                "caption": caption,
                "clean_context": context,
            }

            output_handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    temporary.replace(output_path)

    return {
        "paper_id": folder.name,
        "status": "processed",
        "kept": kept,
        "removed": removed,
        "invalid_json": invalid_json,
        "missing_image": missing_image,
        "short_caption": short_caption,
        "duplicate": duplicate,
    }


def select_papers(
    root: Path,
    paper: str | None,
    start: int,
    end: int | None,
) -> list[Path]:
    if paper:
        return [root / paper]

    folders = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    return folders[start:end]


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    root = args.root.expanduser().resolve()

    if not root.is_dir():
        LOGGER.error("Dataset root does not exist: %s", root)
        return 2

    if args.max_context_chars < 1:
        LOGGER.error("--max-context-chars must be positive.")
        return 2

    papers = select_papers(
        root=root,
        paper=args.paper,
        start=args.start,
        end=args.end,
    )

    LOGGER.info("Selected %d paper folders.", len(papers))

    summaries: list[dict[str, Any]] = []

    for folder in papers:
        if not folder.is_dir():
            LOGGER.warning(
                "Paper directory not found: %s",
                folder,
            )
            continue

        result = process_paper(
            folder=folder,
            input_name=args.input_name,
            output_name=args.output_name,
            max_context_chars=args.max_context_chars,
            drop_missing_images=not args.keep_missing_images,
            overwrite=args.overwrite,
        )

        summaries.append(result)

        if result["status"] == "processed":
            LOGGER.info(
                "%s | kept=%d | removed=%d",
                folder.name,
                result["kept"],
                result["removed"],
            )

    run_summary = {
        "papers_selected": len(papers),
        "papers_processed": sum(
            row["status"] == "processed"
            for row in summaries
        ),
        "papers_skipped_existing": sum(
            row["status"] == "skipped_existing"
            for row in summaries
        ),
        "papers_missing_input": sum(
            row["status"] == "missing_input"
            for row in summaries
        ),
        "total_kept": sum(
            int(row.get("kept", 0))
            for row in summaries
        ),
        "total_removed": sum(
            int(row.get("removed", 0))
            for row in summaries
        ),
        "input_name": args.input_name,
        "output_name": args.output_name,
        "max_context_chars": args.max_context_chars,
        "drop_missing_images": not args.keep_missing_images,
        "papers": summaries,
    }

    summary_path = root / args.summary_name
    temporary_summary = summary_path.with_name(
        summary_path.name + ".tmp"
    )
    temporary_summary.write_text(
        json.dumps(
            run_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)

    LOGGER.info(
        "Preprocessing complete | kept=%d | removed=%d",
        run_summary["total_kept"],
        run_summary["total_removed"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
