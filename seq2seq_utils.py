import json
import os
import argparse
import json
from collections import defaultdict

from datasets import load_dataset

DOC_DOMAIN_SPLIT = "train"
YOUR_DATASETS_SOURCE_DIR = ""  # the root folder of your local `datasets` source code.


def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


def btag(tag, text):  # tag the content
    return "<{}> {}".format(tag, text2line(text))


def load_doc2dial_seq2seq(args):
    doc_dataset = load_dataset(
        "doc2dial.py",
        name="document_domain",
        split=DOC_DOMAIN_SPLIT,
        cache_dir=args.cache_dir,
    )
    dial_dataset = load_dataset(
        "doc2dial.py",
        name="dialogue_domain",
        split=args.split,
        cache_dir=args.cache_dir,
        ignore_verifications=True,
    )
    d_doc = defaultdict(dict)
    for ex in doc_dataset:
        d_doc[ex["doc_id"]]["title"] = ex["title"]
        for d_span in ex["spans"]:
            d_doc[ex["doc_id"]][d_span["id_sp"]] = d_span

    with open("docs.json") as f:
        docs = json.load(f)

    source = []
    target = []
    ids = []
    for ex in dial_dataset:
        doc_id = ex["doc_id"]
        doc = docs[doc_id]
        d_doc_spans = d_doc[doc_id]
        dial_context = []
        for i, turn in enumerate(ex["turns"]):
            if not turn[
                "references"
            ]:  # this task only uses instances and evalutes on the grounded turns.
                continue
            utterance = text2line(turn["utterance"])
            utterance = btag(turn["role"], utterance)
            utterance = " ".join(utterance.split())
            if turn["role"] in args.role:  # if current turn is to predict
                contexts = [
                    btag("last_turn", dial_context[-1].split(" ", 1)[-1])
                ]  # add previous utterance as tagged query context
                contexts.extend(
                    dial_context[::-1]
                )  # add dialog history in reverse order as tagged dialogue context

                contexts = " ".join(contexts)
                contexts = " ".join(contexts.split())
                contexts = contexts + " " + doc["text"]

                reference_content = ""
                for ref in turn["references"]:
                    sp_id = ref["sp_id"]
                    reference_content += " " + d_doc_spans[sp_id]["text_sp"]
                reference_context = btag("grounding", reference_content)
                reference_context = " ".join(reference_context.split())

                if args.split == "train":
                    if "1" in args.task:
                        source.append("generate <grounding> then <agent>: " + contexts)
                        target.append(reference_context + " " + utterance)
                        ids.append("{}_{}_1".format(ex["dial_id"], turn["turn_id"] - 1))
                    if "2" in args.task:
                        source.append("generate <grounding>: " + contexts)
                        target.append(reference_context)
                        ids.append("{}_{}_2".format(ex["dial_id"], turn["turn_id"] - 1))
                    if "3" in args.task:
                        source.append("generate <agent>: " + contexts)
                        target.append(utterance)
                        ids.append("{}_{}_3".format(ex["dial_id"], turn["turn_id"] - 1))
                else:
                    source.append("generate <grounding> then <agent>: " + contexts)
                    target.append(reference_context + " " + utterance)
                    ids.append("{}_{}_1".format(ex["dial_id"], turn["turn_id"] - 1))

            dial_context.append(utterance)

    assert len(source) == len(
        target
    ), "Need to ensure that source and target are same sized."
    if args.split == "validation":
        args.split = "val"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(
        os.path.join(args.output_dir, "{}.source".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(source))
        fp.close()
    with open(
        os.path.join(args.output_dir, "{}.target".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(target))
        fp.close()
    with open(
        os.path.join(args.output_dir, "{}.ids".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(ids))
        fp.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Data split is 'train', 'validation' or 'test'",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="agent",
        help="which role's utterance for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output the data files",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="1",
        help="Which task",
    )

    args = parser.parse_args()
    load_doc2dial_seq2seq(args)


if __name__ == "__main__":
    main()
