import os
import argparse
import json

from datasets import load_dataset

DOC_DOMAIN_SPLIT = "train"


def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


def btag(tag, text):  # tag the content
    return "<{}>{}</{}>".format(tag, text2line(text), tag)


def process_doc(args):
    doc_dataset = load_dataset(
        "doc2dial.py",
        name="document_domain",
        split=DOC_DOMAIN_SPLIT,
        cache_dir=args.cache_dir,
    )

    d_doc = {}
    for ex in doc_dataset:
        doc_id = ex["doc_id"]
        d_doc[doc_id] = {}
        doc_title = btag("title", ex["title"].split("#")[0])
        spans_text = []
        for d_span in ex["spans"]:
            tag = d_span["tag"]
            text_sp = d_span["text_sp"]

            if tag != "u":
                spans_text.append(btag(tag, text2line(text_sp)))
            else:
                spans_text.append(text2line(text_sp))

        d_doc[doc_id]["text"] = " ".join([doc_title] + spans_text)

    with open(os.path.join(args.output_dir, "docs.json"), "w") as f:
        json.dump(d_doc, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output file",
    )

    args = parser.parse_args()
    process_doc(args)


if __name__ == "__main__":
    main()
