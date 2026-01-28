import sys
from pathlib import Path
import argparse
import json
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "src"))

# reuse the ranking function from the existing script
from src.scripts.rank_from_pdfs import rank_job_against_texts
from src.core.cleaning import clean_text_pipeline

def main(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--text", nargs="*", help="paths to plain-text resume files (.txt). Accepts multiple")
    parser.add_argument("--text-dir", help="directory with .txt resumes (recursive)")
    parser.add_argument("--bulk-file", help="single file containing multiple resumes separated by a delimiter (see --delimiter)")
    parser.add_argument("--delimiter", default="---", help="delimiter string used inside bulk-file to separate resumes (default: '---')")
    parser.add_argument("--skills", help="comma-separated required skills (optional)")
    parser.add_argument(
        "--debug-text",
        action="store_true",
        help="Print raw and cleaned text from .txt files, then exit")
    args = parser.parse_args(argv)

    text_paths: List[Path] = []
    if args.text:
        text_paths += [Path(p) for p in args.text]
    if args.text_dir:
        text_paths += list(Path(args.text_dir).expanduser().rglob("*.txt"))

    texts: List[str] = []
    ids: List[str] = []

    # read individual files
    for p in text_paths:
        try:
            raw_text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw_text = ""
        cleaned_text = clean_text_pipeline(raw_text)
        texts.append(cleaned_text)
        ids.append(p.name)

        if args.debug_text:
            print("\n" + "=" * 80)
            print(f"FILE: {p.name}")
            print("-" * 80)
            print("RAW TEXT (first 1500 chars):\n")
            print(raw_text[:1500])
            print("\n" + "-" * 80)
            print("CLEANED TEXT (first 1500 chars):\n")
            print(cleaned_text[:1500])
            print("\n" + "=" * 80)

    # read bulk file (one file containing many resumes separated by delimiter)
    if args.bulk_file:
        bf = Path(args.bulk_file)
        if bf.exists():
            try:
                bulk_raw = bf.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                bulk_raw = ""
            if bulk_raw:
                parts = [p.strip() for p in bulk_raw.split(args.delimiter) if p.strip()]
                for i, part in enumerate(parts):
                    cleaned = clean_text_pipeline(part)
                    texts.append(cleaned)
                    ids.append(f"{bf.name}::part_{i+1}")
                    if args.debug_text:
                        print("\n" + "=" * 80)
                        print(f"BULK FILE: {bf.name} (part {i+1})")
                        print("-" * 80)
                        print("RAW PART (first 1500 chars):\n")
                        print(part[:1500])
                        print("\n" + "-" * 80)
                        print("CLEANED PART (first 1500 chars):\n")
                        print(cleaned[:1500])
                        print("\n" + "=" * 80)
        else:
            print(f"Warning: bulk file not found: {bf}", file=sys.stderr)

    if not texts:
        print("No text resumes provided. Use --text <file1.txt> [file2.txt ...], --text-dir <dir>, or --bulk-file <file>")
        sys.exit(1)

    if args.debug_text:
        print("\n✅ Text extraction & cleaning completed successfully.")
        sys.exit(0)

    skills = [s.strip() for s in args.skills.split(",")] if args.skills else None

    out = rank_job_against_texts(args.title, args.description, texts, ids, required_skills=skills)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
# ...existing code...



# import sys
# from pathlib import Path
# import argparse
# import json
# from typing import List, Optional

# ROOT = Path(__file__).resolve().parents[1]  # project root
# sys.path.insert(0, str(ROOT / "src"))

# # reuse the ranking function from the existing script
# from src.scripts.rank_from_pdfs import rank_job_against_texts
# from src.core.cleaning import clean_text_pipeline

# def main(argv: List[str]):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--title", required=True)
#     parser.add_argument("--description", required=True)
#     parser.add_argument("--text", nargs="*", help="paths to plain-text resume files (.txt)")
#     parser.add_argument("--text-dir", help="directory with .txt resumes (recursive)")
#     parser.add_argument("--skills", help="comma-separated required skills (optional)")
#     parser.add_argument(
#         "--debug-text",
#         action="store_true",
#         help="Print raw and cleaned text from .txt files, then exit")
#     args = parser.parse_args(argv)

#     text_paths: List[Path] = []
#     if args.text:
#         text_paths += [Path(p) for p in args.text]
#     if args.text_dir:
#         text_paths += list(Path(args.text_dir).expanduser().rglob("*.txt"))

#     if not text_paths:
#         print("No text files provided. Use --text <file1.txt> or --text-dir <dir>")
#         sys.exit(1)

#     texts = []
#     ids = []
#     for p in text_paths:
#         try:
#             raw_text = p.read_text(encoding="utf-8", errors="ignore")
#         except Exception:
#             raw_text = ""
#         cleaned_text = clean_text_pipeline(raw_text)

#         texts.append(raw_text)
#         ids.append(p.name)

#         if args.debug_text:
#             print("\n" + "=" * 80)
#             print(f"FILE: {p.name}")
#             print("-" * 80)
#             print("RAW TEXT (first 1500 chars):\n")
#             print(raw_text[:1500])
#             print("\n" + "-" * 80)
#             print("CLEANED TEXT (first 1500 chars):\n")
#             print(cleaned_text[:1500])
#             print("\n" + "=" * 80)

#     if args.debug_text:
#         print("\n✅ Text extraction & cleaning completed successfully.")
#         sys.exit(0)

#     skills = [s.strip() for s in args.skills.split(",")] if args.skills else None

#     out = rank_job_against_texts(args.title, args.description, texts, ids, required_skills=skills)
#     print(json.dumps(out, indent=2))

# if __name__ == "__main__":
#     main(sys.argv[1:])