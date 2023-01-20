from utils import (
    load_inputs,
    load_good_induction_candidates,
    replace_toks_by_frequency_rank,
    DATA_PATH,
)

import pickle
import os.path
import torch
from tqdm import tqdm
from collections import defaultdict


def build_basic_token_masks():
    inps = load_inputs()[:, :-1]
    good_induction_candidates = load_good_induction_candidates().to(dtype=torch.bool)

    candidates_mask = torch.zeros_like(inps, dtype=torch.bool)
    repeats_mask = torch.zeros_like(inps, dtype=torch.bool)
    for i, row in tqdm(enumerate(inps), total=inps.shape[0]):
        seen_toks = set()
        for j, tok in enumerate(row):
            if good_induction_candidates[tok]:
                candidates_mask[i, j] = True
            if tok.item() in seen_toks:
                repeats_mask[i, j] = True
            seen_toks.add(tok.item())
    with open(os.path.join(DATA_PATH, "mask_candidates.pkl"), "wb") as f:
        pickle.dump(candidates_mask, f)
    with open(os.path.join(DATA_PATH, "mask_repeats.pkl"), "wb") as f:
        pickle.dump(repeats_mask, f)
    with open(os.path.join(DATA_PATH, "mask_repeat_candidates.pkl"), "wb") as f:
        pickle.dump(repeats_mask.logical_and(candidates_mask), f)


def build_fuzzy_repeat_mask(stdizing_func, mask_name="fuzzy_repeats"):
    """
    stdizing_func is a function that takes in a single token index and returns
    a "canonical form" for that token index. The idea is that this should be a
    non-injective function and partition the vocabulary into equivalence
    classes. This function then builds a repeats' mask where occurrences of an
    already-represented equivalence class count as repeats in that example.

    The output of stdizing_func is assumed hashable.
    """
    inps = load_inputs()[:, :-1]

    repeats_mask = torch.zeros_like(inps, dtype=torch.bool)
    for i, row in tqdm(enumerate(inps), total=inps.shape[0]):
        seen_toks = set()
        for j, tok in enumerate(row):
            stdized_tok = stdizing_func(tok)
            if stdized_tok in seen_toks:
                repeats_mask[i, j] = True
            seen_toks.add(stdized_tok)
    with open(os.path.join(DATA_PATH, f"mask_{mask_name}.pkl"), "wb") as f:
        pickle.dump(repeats_mask, f)


def build_token_frequency_mask(top_k):
    inps = load_inputs()[:, :-1]
    inps_by_frequency = replace_toks_by_frequency_rank(inps)
    return inps_by_frequency >= top_k


def build_end_of_repeated_bigram_mask(stdizing_func=lambda x: x, mask_name="ends_of_repeated_bigrams"):
    """
    Pickle and return a bool tensor of shape 104091 x 301 (dataset without index
    column) indicating whether the given dataset token is the end of a repeated
    bigram in that example (i.e. whether performing strict induction on the
    previous token would upweigh the given token).

    If stdizing_func is provided, each token is first transformed by
    stdizing_func before being evaluated as a repeat or not. The point is for
    stdizing_func to be a non-injective function with hashable outputs that
    maps similar tokens to the same canonical form, so that we may evaluate
    fuzzy repetition.
    """
    inps = load_inputs()[:, :-1]
    end_of_repeated_bigram_mask = torch.zeros_like(inps, dtype=torch.bool)
    for i, row in tqdm(enumerate(inps), total=inps.shape[0]):
        fst = stdizing_func(row[0].item())
        snd = stdizing_func(row[1].item())
        seen_bigrams = set([(fst, snd)])
        for j, tok in enumerate(row[2:]):
            fst, snd = snd, stdizing_func(tok.item())
            if (fst, snd) in seen_bigrams:
                end_of_repeated_bigram_mask[i, j+2] = True
            else:
                seen_bigrams.add((fst, snd))
    with open(os.path.join(DATA_PATH, f"mask_{mask_name}.pkl"), "wb") as f:
        pickle.dump(end_of_repeated_bigram_mask, f)


def build_misleading_induction_mask():
    inps = load_inputs()[:, :-1]
    misleading_induction_mask = torch.zeros_like(inps, dtype=torch.bool)
    for i, row in tqdm(enumerate(inps), total=inps.shape[0]):
        fst = row[0].item()
        snd = row[1].item()
        seen_bigrams = defaultdict(set)
        seen_bigrams[fst].add(snd)
        for j, tok in enumerate(row[2:]):
            fst, snd = snd, tok.item()
            if fst in seen_bigrams and snd not in seen_bigrams[fst]:
                misleading_induction_mask[i, j+2] = True
            seen_bigrams[fst].add(snd)
    with open(os.path.join(DATA_PATH, "mask_misleading_induction.pkl"), "wb") as f:
        pickle.dump(misleading_induction_mask, f)


def get_all_masks(inp_ixes=None):
    if inp_ixes is None:
        inp_ixes = load_inputs()[:, -1]
    
    with open(os.path.join(DATA_PATH, "mask_candidates.pkl"), "rb") as f:
        induction_candidates = pickle.load(f)[inp_ixes][:, :-1]

    with open(os.path.join(DATA_PATH, "mask_repeat_candidates.pkl"), "rb") as f:
        repeat_candidates = pickle.load(f)[inp_ixes][:, :-1]

    with open(os.path.join(DATA_PATH, "mask_repeats.pkl"), "rb") as f:
        repeats = pickle.load(f)[inp_ixes][:, 1:]

    untop_200 = build_token_frequency_mask(200)[inp_ixes]
    uncommon_repeats = repeats.logical_and(untop_200[:, 1:])

    with open(os.path.join(DATA_PATH, "mask_ends_of_repeated_bigrams.pkl"), "rb") as f:
        erb = pickle.load(f)[inp_ixes][:, 1:]

    nerb = erb.logical_not()
    candidate_erb = induction_candidates.logical_and(erb)
    nerb_uncommon_repeats = nerb.logical_and(uncommon_repeats)

    with open(os.path.join(DATA_PATH, "mask_simple_strip_fuzzy_ends_of_repeated_bigrams.pkl"), "rb") as f:
        ferb = pickle.load(f)[inp_ixes][:, 1:]

    nferb = ferb.logical_not()
    ferbnerb = ferb.logical_and(nerb)
    candidate_ferb = induction_candidates.logical_and(ferb)
    candidate_ferbnerb = induction_candidates.logical_and(ferbnerb)
    nferb_uncommon_repeats = nferb.logical_and(uncommon_repeats)
    nferbnerb_uncommon_repeats = ferbnerb.logical_and(uncommon_repeats).logical_not()


    with open(os.path.join(DATA_PATH, "mask_misleading_induction.pkl"), "rb") as f:
        misleading_induction = pickle.load(f)[inp_ixes]
    misleading_induction = misleading_induction[:, 1:].logical_and(untop_200[:, :-1])

    return {
        "induction_candidates"       : induction_candidates,
        "repeat_candidates"          : repeat_candidates,
        "repeats"                    : repeats,
        "uncommon_repeats"           : uncommon_repeats,
        "erb"                        : erb,
        "ferb"                       : ferb,
        "ferbnerb"                   : ferbnerb,
        "candidate_erb"              : candidate_erb,
        "candidate_ferb"             : candidate_ferb,
        "candidate_ferbnerb"         : candidate_ferbnerb,
        "nerb_uncommon_repeats"      : nerb_uncommon_repeats,
        "nferb_uncommon_repeats"     : nferb_uncommon_repeats,
        "nferbnerb_uncommon_repeats" : nferbnerb_uncommon_repeats,
        "misleading_induction"       : misleading_induction,
    }