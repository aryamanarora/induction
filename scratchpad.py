# %%
import torch

good_induction_candidates = torch.tensor([0, 1, 0, 1, 1])

a = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 1], [0, 1, 2, 1, 1], [0, 0, 0, 0, 0], [0, 1, 2, 2, 1]])


def get_induction_candidate_mask(
    t: torch.Tensor, good_induction_candidates: torch.Tensor, match_all_occurrences=False
) -> torch.Tensor:
    """
    t is a 2d Tensor of token indices of size batch_size x seq_len
    good_induction_candidate is a 1d Tensor of 0s and 1s of size vocab_size indicating whether the ith token is a good induction candidate in general
    Return a 2d Tensor of bools indicating whether each token in t is a repeated occurrence of a good induction candidate in that row (or, if match_all_occurrences is True, we also set the first occurrence of the token to True)
    """
    res = torch.ones_like(t, dtype=torch.bool)
    good_induction_candidates = good_induction_candidates.to(dtype=torch.bool)
    # Sorry, couldn't find anything better than a double-for
    for i, row in enumerate(t):
        seen_toks = set()
        for j, tok in enumerate(row):
            if tok.item() in seen_toks or match_all_occurrences:
                res[i, j] = good_induction_candidates[tok]
            else:
                seen_toks.add(tok.item())
                res[i, j] = False

    return res


def decode_and_highlight(toks_to_decode, highlight_ixes, tokenizer):
    if len(toks_to_decode.shape) == 1:
        toks_to_decode = toks_to_decode.unsqueeze(0)
    res = []
    for row in toks_to_decode:
        row_toks = []
        for i, ch in enumerate(row):
            if highlight_ixes[ch]:
                row_toks += [58, 58, 58]
                row_toks.append(ch)
                row_toks += [60, 60, 60]
            else:
                row_toks.append(ch)
        res.append("".join(tokenizer.batch_decode(row_toks)))
    return res


# %%
get_induction_candidate_mask(a, good_induction_candidates, match_all_occurrences=True)
# %%
