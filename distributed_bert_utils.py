
# import pandas as pd
import torch


# def get_word(token):
#     return [x for x, y in vocab.items() if y == token]


def get_vocab_sorted_dict(trainer):
    from transformers import PreTrainedTokenizer

    vocab = trainer.tokenizer.get_vocab()

    freq = {vocab[token]: 0 for token in vocab}
    for x in trainer.train_dataset:
        for w in x['input_ids'][1:-1]:
            freq[w] += 1

    del freq[trainer.tokenizer.mask_token_id]
    del freq[trainer.tokenizer.sep_token_id]
    del freq[trainer.tokenizer.cls_token_id]
    del freq[trainer.tokenizer.unk_token_id]
    del freq[trainer.tokenizer.pad_token_id]

    hist = [freq[x] for x in freq]
    hist.sort()
    hist.reverse()

    sorted_dict = dict(sorted(freq.items(), key=lambda item: -item[1]))

    # df = pd.DataFrame(hist[20:1000])
    # df = df.rename(columns={0: 'frequency'})
    # df.reset_index().plot.scatter(x='index', y='frequency', ylim=(0, 500), figsize=(15,5))

    return sorted_dict


def get_vocab_distribution(sorted_vocab_dict, mems, failure_probs, smoothing_freq=1, distribution_cut=2000, use_heuristic=True):
    W = torch.tensor(list(sorted_vocab_dict.values()), dtype=torch.float32)[distribution_cut:]
    W += smoothing_freq
    T = len(W)
    D = len(mems)
    alpha = torch.ones(T)

    A = torch.zeros((T, D))

    for i in range(D):
        
        P = W * alpha
        P /= P.sum()
        P = torch.clamp(P*mems[i], min=0, max=1.0)

        while P.sum() < 0.99 * mems[i]:
            P = torch.clamp(P*1.01, min=0, max=1.0)
            if P[-1] >= 0.99:
                break
        nm = 0
        if not use_heuristic:
            P = torch.ones_like(P)/P.shape[0]*mems[i]

        placement = torch.bernoulli(P)

        while placement.sum() > mems[i]:
            placement = torch.bernoulli(P)
        
        A[:,i] += placement
        alpha[placement > 0.5] *= (failure_probs[i]+0.01)

        # print('------')
    return torch.tensor(list(sorted_vocab_dict.keys()))[distribution_cut:], A
