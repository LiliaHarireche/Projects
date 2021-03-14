"""

"""
# External imports
from pprint import pprint
from collections import defaultdict, Counter
from operator import itemgetter
from typing import List
import csv

# Local imports
from languages.spacy_support import SpacyPredictor
from languages.german import GermanPredictor
from languages.pymorph_support import PymorphPredictor
from languages.semitic import HebrewPredictor, ArabicPredictor
from languages.util import GENDER, SPACY_GENDER_TYPES
from evaluation import evaluate_bias

LANGUAGE_PREDICTOR = {
    "es": lambda: SpacyPredictor("es"),
    "fr": lambda: SpacyPredictor("fr"),
    "it": lambda: SpacyPredictor("it"),
    "ru": lambda: PymorphPredictor("ru"),
    "uk": lambda: PymorphPredictor("uk"),
    "he": lambda: HebrewPredictor(),
    "ar": lambda: ArabicPredictor(),
    "de": lambda: GermanPredictor()
}


def get_src_indices(instance: List[str]) -> List[int]:
    """
    (English)
    Determine a list of source side indices pertaining to a
    given instance in the dataset.
    """
    _, src_word_ind, sent = instance[: 3]  # cf data/aggregates
    src_word_ind = int(src_word_ind)
    sent_tok = sent.split(" ")
    if (src_word_ind > 0) and (sent_tok[src_word_ind - 1].lower() in ["the", "an", "a"]):
        src_indices = [src_word_ind - 1]
    else:
        src_indices = []
    src_indices.append(src_word_ind)

    return src_indices


def get_translated_professions(alignment_fn, ds: List[List[str]], bitext: List[List[str]]) -> List[str]:  # noqa
    """
    (Language independent)
    Load alignments from file and return the translated profession according to
    source indices.
    """
    # Load files and data structures
    ds_src_sents = list(map(itemgetter(2), ds))
    bitext_src_sents = [src_sent for ind, (src_sent, tgt_sent) in bitext]

    # Sanity checks
    assert len(ds) == len(bitext)
    mismatched = [ind for (ind, (ds_src_sent, bitext_src_sent)) in enumerate(zip(ds_src_sents, bitext_src_sents))
                  if ds_src_sent != bitext_src_sent]
    if len(mismatched) != 0:
        raise AssertionError  # end

    bitext = [(ind, (src_sent.split(), tgt_sent.split()))
              for ind, (src_sent, tgt_sent) in bitext]

    src_indices = list(map(get_src_indices, ds))

    if isinstance(alignment_fn, str):
        full_alignments = []
        for line in open(alignment_fn):
            cur_align = defaultdict(list)
            for word in line.split():
                src, tgt = word.split("-")
                cur_align[int(src)].append(int(tgt))
            full_alignments.append(cur_align)
    elif isinstance(alignment_fn, list):
        full_alignments = []
        for line in alignment_fn:
            cur_align = defaultdict(list)
            for word in line.split():
                src, tgt = word.split("-")
                cur_align[int(src)].append(int(tgt))
            full_alignments.append(cur_align)
    else:
        raise TypeError('Unknown type for alignment_fn, you may use a string (path) or list')

    bitext_inds = [ind for ind, _ in bitext]

    alignments = []
    for ind in bitext_inds:
        alignments.append(full_alignments[ind])

    assert len(bitext) == len(alignments)
    assert len(src_indices) == len(alignments)

    translated_professions = []
    target_indices = []

    for (_, (src_sent, tgt_sent)), alignment, cur_indices in zip(bitext, alignments, src_indices):
        # cur_translated_profession = " ".join([tgt_sent[cur_tgt_ind]
        #                                       for src_ind in cur_indices
        #                                       for cur_tgt_ind in alignment[src_ind]])
        cur_tgt_inds = ([cur_tgt_ind
                         for src_ind in cur_indices
                         for cur_tgt_ind in alignment[src_ind]])

        cur_translated_profession = " ".join([tgt_sent[cur_tgt_ind]
                                              for cur_tgt_ind in cur_tgt_inds])
        target_indices.append(cur_tgt_inds)
        translated_professions.append(cur_translated_profession)

    return translated_professions, target_indices


def align_bitext_to_ds(bitext, ds):
    """
    Return a subset of bitext that's aligned to ds.
    """
    bitext_dict = dict([(src, (ind, tgt)) for ind, (src, tgt) in enumerate(bitext)])
    new_bitext = []
    for entry in ds:
        en_sent = entry[2]
        ind, tgt_sent = bitext_dict[en_sent]
        new_bitext.append((ind, (en_sent, tgt_sent)))
    return new_bitext


if __name__ == "__main__":
    lang = 'it'
    gender_predictor = LANGUAGE_PREDICTOR[lang]()

    ds_fn = '../data/en.txt'
    bi_fn = '../translations/aws/en-{}.txt'.format(lang)
    al_fn = '../translations/_aligns/aws/forwarden-{}.align'.format(lang)

    ds = [line.strip().split("\t") for line in open(ds_fn, encoding="utf8")]  # ds_fn == en.txt
    full_bitext = [line.strip().split(" ||| ") for line in open(bi_fn, encoding="utf8")]  # bi_fn == translate file

    bitext = align_bitext_to_ds(full_bitext, ds)

    translated_profs, tgt_inds = get_translated_professions(al_fn, ds, bitext)
    assert (len(translated_profs) == len(tgt_inds))

    target_sentences = [tgt_sent for (ind, (src_sent, tgt_sent)) in bitext]

    gender_predictions = [gender_predictor.get_gender(prof, translated_sent, entity_index, ds_entry)
                          for prof, translated_sent, entity_index, ds_entry
                          in zip(translated_profs,
                                 target_sentences,
                                 map(lambda ls: min(ls, default=-1), tgt_inds),
                                 ds)]

    d = evaluate_bias(ds, gender_predictions)
    pprint(d)
