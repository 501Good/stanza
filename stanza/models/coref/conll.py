"""Contains functions to produce conll-formatted output files with
predicted spans and their clustering"""

from collections import defaultdict
from contextlib import contextmanager
import os
from typing import List, TextIO

from stanza.models.coref.config import Config
from stanza.models.coref.const import Doc, Span


# pylint: disable=too-many-locals
def write_conll(doc: Doc, clusters: List[List[Span]], heads: List[int], f_obj: TextIO):
    """Writes span/cluster information to f_obj, which is assumed to be a file
    object open for writing"""
    placeholder = list("\t_" * 7)
    # the nth token needs to be a number
    placeholder[9] = "0"
    placeholder = "".join(placeholder)
    doc_id = doc["document_id"]
    words = doc["cased_words"]
    part_id = doc["part_id"]
    sents = doc["sent_id"]
    mwts = doc["mwts"]
    head = doc["head"]
    deprel = doc["deprel"]
    sent_id_name = doc["sent_id_name"]

    max_word_len = max(len(w) for w in words)

    starts = defaultdict(lambda: [])
    ends = defaultdict(lambda: [])
    single_word = defaultdict(lambda: [])

    for cluster_id, cluster in enumerate(clusters):
        if len(heads[cluster_id]) != len(cluster):
            # TODO debug this fact and why it occurs
            # print(f"cluster {cluster_id} doesn't have the same number of elements for word and span levels, skipping...")
            continue
        for cluster_part, (start, end) in enumerate(cluster):
            if end - start == 1:
                single_word[start].append((cluster_part, cluster_id))
            else:
                starts[start].append((cluster_part, cluster_id))
                ends[end - 1].append((cluster_part, cluster_id))

    f_obj.write(f"# newdoc id = {doc_id}\n# global.Entity = eid-head")

    total_words = 0
    word_number = 0
    sent_id = 0
    for word_id, word in enumerate(words):
        cluster_info_lst = []
        for part, cluster_marker in starts[word_id]:
            start, end = clusters[cluster_marker][part]
            mention_head = min(heads[cluster_marker][part] - start + 1, end - start)
            cluster_info_lst.append(f"(e{part_id}.{cluster_marker}-{mention_head}")
        for part, cluster_marker in single_word[word_id]:
            start, end = clusters[cluster_marker][part]
            mention_head = min(heads[cluster_marker][part] - start + 1, end - start)
            cluster_info_lst.append(f"(e{part_id}.{cluster_marker}-{mention_head})")
        for part, cluster_marker in ends[word_id]:
            cluster_info_lst.append(f"e{part_id}.{cluster_marker})")


        # we need our clusters to be ordered such that the one that is closest the first change
        # is listed last in the chains
        def compare_sort(x):
            split = x.split("-")
            if len(split) > 1:
                return int(split[-1].replace(")", "").strip())
            else:
                # we want everything that's a closer to be first
                return float("inf")

        cluster_info_lst = sorted(cluster_info_lst, key=compare_sort, reverse=True)
        cluster_info = "".join(cluster_info_lst) if cluster_info_lst else "_"

        if word_id == 0 or sents[word_id] != sents[word_id - 1]:
            # f_obj.write(f"\n# sent_id = {doc_id}.{sent_id}\n")
            f_obj.write(f"\n# sent_id = {sent_id_name[sent_id]}\n")
            total_words += word_number - 1 if word_id > 0 else 0
            word_number = 1
            sent_id += 1

        if str(word_id) in mwts:
            mwt = mwts[str(word_id)]
            mwt_id = f"{word_number}-{word_number + mwt['len'] - 1}"
            f_obj.write(f"{mwt_id}\t{mwt['text']}\t{'\t'.join(['_'] * 8)}\n")

        if cluster_info != "_":
            cluster_info = f"Entity={cluster_info}"

        head_id = 0 if head[word_id] == "null" else head[word_id] - total_words + 1

        f_obj.write(f"{word_number}\t{word}\t_\t_\t_\t_\t{head_id}\t{deprel[word_id]}\t_\t{cluster_info}\n")

        word_number += 1

    f_obj.write("\n")


@contextmanager
def open_(config: Config, epochs: int, data_split: str):
    """Opens conll log files for writing in a safe way."""
    base_filename = f"{config.section}_{data_split}_e{epochs}"
    conll_dir = config.conll_log_dir
    kwargs = {"mode": "w", "encoding": "utf8"}

    os.makedirs(conll_dir, exist_ok=True)

    with open(
        os.path.join(  # type: ignore
            conll_dir, f"{base_filename}.gold.conll"
        ),
        **kwargs,
    ) as gold_f:
        with open(
            os.path.join(  # type: ignore
                conll_dir, f"{base_filename}.pred.conll"
            ),
            **kwargs,
        ) as pred_f:
            yield (gold_f, pred_f)
