from jaxtyping import Float
from typing import Optional, Union, List, Literal, Tuple
import torch as t
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
import einops
from pathlib import Path
import webbrowser
from IPython.display import display, HTML

from circuitsvis.attention import attention_heads, attention_patterns


def concat_lists(*lists):
    return [x for l in lists for x in l]


def diagonalise(tensor: Tensor, dim: int):
    '''
    Returns a tensor with dimension `dim` duplicated, diagonal elements along these two dims
    are given by `tensor`, and off-diags are zero.
    '''
    shape = tensor.shape
    new_shape = shape[:dim] + (shape[dim],) + shape[dim:]
    new_tensor = t.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    indices = concat_lists(
        [slice(None)] * dim,
        [range(shape[dim]), range(shape[dim])],
        [slice(None)] * (len(shape) - dim - 1)
    )
    new_tensor[indices] = tensor
    return new_tensor


def from_cache(
    cache: ActivationCache,
    model: HookedTransformer,
    tokens: Union[List[str], Float[Tensor, "seq"]],
    resid_directions: Optional[Union[Float[Tensor, "*seq d_model"], Float[Tensor, "*seq d_vocab"]]] = None,
    seq_pos: Optional[Union[int, List[int]]] = None,
    layers: Optional[Union[int, List[int]]] = None,
    heads: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
    mode: Literal["large", "small"] = "small",
    return_mode: Literal["html", "browser", "view"] = "html",
    head_notation: Literal["dot", "LH"] = "dot",
    include_b_U_attribution: bool = False, # This always points towards more common tokens I guess, so it does a lot of the heavy lifting for us (when we aren't using a baseline). Interesting!
):
    '''
    resid_directions
        if not None, then these are the directions we do attribution with (unless it has length d_vocab, in which case this is interpreted as something like logit_diff, i.e. it might have 1s and -1s in the elements).
    tokens
        if not None, then we use these unembeddings as our resid_directions
    seq_pos
        if not None, then we only do attribution for this position (set all other
        positions to be zero)
    '''
    t.cuda.empty_cache()
    assert not(cache.has_batch_dim), "Only supports batch dim = 1 (otherwise things get too big and messy!)"
    assert return_mode in ["html", "browser", "view"], "return_mode must be one of 'html', 'browser' or 'view'"
    seq_len = len(tokens)

    if seq_pos is None:
        seq_pos = list(range(seq_len))
    elif isinstance(seq_pos, int):
        seq_pos = [seq_pos]
    assert isinstance(seq_pos, list) and all(isinstance(i, int) for i in seq_pos), "seq_pos must be None, int, or list of ints"

    # Get the layers we'll be using (or the heads we'll be using)
    if (layers is None) and (heads is None):
        layers = list(range(model.cfg.n_layers))
        heads = [(layer, head) for layer in layers for head in range(model.cfg.n_heads)]
    elif (layers is None) and (heads is not None):
        heads = [heads] if isinstance(heads[0], int) else heads
        heads = [(layer % model.cfg.n_layers, head % model.cfg.n_heads) for layer, head in heads]
        layers = sorted(set(layer for layer, head in heads))
    elif (layers is not None) and (heads is None):
        layers = [layers] if isinstance(layers, int) else layers
        layers = [layer % model.cfg.n_layers for layer in layers]
        heads = [(layer, head) for layer in layers for head in range(model.cfg.n_heads)]
    else:
        raise ValueError("Can only specify layers or heads, not both.")

    # ! Get MLP & other decomps
    embed_results: Float[Tensor, "2 seqQ d_model"] = t.stack([
        cache["embed"], cache["pos_embed"]
    ]) / cache["scale"]
    embed_results: Float[Tensor, "2 seqQ seqK d_model"] = diagonalise(embed_results, dim=1)[:, seq_pos]
    mlp_results: Float[Tensor, "layer seqQ d_model"] = t.stack([
        cache["mlp_out", layer] for layer in layers
    ]) / cache["scale"]
    mlp_results: Float[Tensor, "layer seqQ seqK d_model"] = diagonalise(mlp_results, dim=1)[:, seq_pos]
    mlp_labels = [f"MLP{layer}" for layer in layers]

    # ! Get attention biases
    attn_biases: Float[Tensor, "layer seqQ seqK d_model"] = diagonalise(
        einops.repeat(model.b_O[layers], "layer d_model -> layer seqQ d_model", seqQ=seq_len), dim=1
    )[:, seq_pos] / cache["scale"][seq_pos]
    attn_bias_labels = [f"Attn bias [{layer}]" for layer in layers]

    # ! Get attention decomposition (this is harder because we have to decompose by source position)
    attn_results = []
    attn_labels = []
    # TODO - could save memory if I didn't have things with `d_model` dimension much; I multiply along this straight away not @ end
    for (layer, head) in heads:
        pattern = cache["pattern", layer][head, seq_pos] # [seqQ seqK]
        v = cache["v", layer][:, head] # [seqK d_head]
        v_post = einops.einsum(
            v, model.W_O[layer, head],
            "seqK d_head, d_head d_model -> seqK d_model",
        )
        results_pre = einops.einsum(
            v_post, pattern,
            "seqK d_model, seqQ seqK -> seqQ seqK d_model",
        )
        # Apply final layernorm (needs to be by query position, not by key position)
        results_pre = results_pre / einops.repeat(
            cache["scale"][seq_pos],
            "seqQ d_model -> seqQ seqK d_model", seqK=seq_len
        )
        attn_results.append(results_pre)
        attn_labels.append(f"{layer}.{head}" if (head_notation == "dot") else f"L{layer}H{head}")

    labels = ["embed", "pos_embed"] + mlp_labels + attn_bias_labels + attn_labels
    full_decomp: Float[Tensor, "component seqQ seqK d_model"] = t.cat([
        embed_results, mlp_results, attn_biases, t.stack(attn_results)
    ])

    # Get the residual stream directions
    if isinstance(tokens[0], str):
        token_ids = model.to_tokens(tokens, prepend_bos=False).squeeze()
        token_str = tokens
    else:
        token_ids = tokens
        token_str = model.to_str_tokens(tokens)
    
    if resid_directions is None:
        assert seq_pos == list(range(seq_len)), "If resid_directions is None, then seq_pos must be None (we're doing attribution per token, in the direction of the correct token's unembedding)."
        seq_pos = list(range(seq_len - 1))
        resid_directions: Float[Tensor, "seqQ d_model"] = model.W_U.T[token_ids[1:]]
        full_decomp: Float[Tensor, "component seqQ seqK d_model"] = full_decomp[:, :-1, :-1]
        b_U_attribution: Float[Tensor, "seqQ"] = model.b_U[token_ids[1:]]
        b_U_attribution: Float[Tensor, "1 seqQ seqK"] = t.diag(b_U_attribution)[seq_pos].unsqueeze(0)
        token_str = token_str[:-1]
    else:
        if resid_directions.ndim == 1:
            resid_directions = einops.repeat(resid_directions, "d -> seqQ d", seqQ=len(seq_pos))
        assert (resid_directions.ndim == 2) and (resid_directions.size(0) <= seq_len)

        if resid_directions.size(1) == model.cfg.d_model:
            b_U_attribution = None
        elif resid_directions.size(1) == model.cfg.d_vocab:
            # In this case, we need bias attribution, and we need to redefine resid directions
            b_U_attribution: Float[Tensor, "1 seqQ seqK"] = t.zeros(1, len(seq_pos), seq_len, device=model.b_U.device)
            b_U_attribution[:, :, seq_pos] = einops.einsum(
                model.b_U, resid_directions,
                "d_vocab, seqQ d_vocab -> seqQ"
            )
            resid_directions = einops.einsum(
                model.W_U, resid_directions,
                "d_model d_vocab, seq d_vocab -> seq d_model"
            )
        else:
            raise ValueError(f"resid_directions must have shape (*seq_len, d_model) or (*seq_len, d_vocab), but the last dimension doesn't match. Shape is {resid_directions.shape}")

    full_attribution = einops.einsum(
        full_decomp, resid_directions,
        "component seqQ seqK d_model, seqQ d_model -> component seqQ seqK"
    )
    if (b_U_attribution is not None) and include_b_U_attribution:
        full_attribution = t.concat([full_attribution, b_U_attribution])
        labels.append("b_U")

    full_attribution_max = einops.reduce(full_attribution.abs(), "c sQ sK -> 1 1 1", "max") # or 1 sQ 1
    full_attribution_scaled_positive = (full_attribution * (full_attribution > 0).float()) / full_attribution_max
    full_attribution_scaled_negative = (-full_attribution * (full_attribution < 0).float()) / full_attribution_max

    # Now finally, we (annoyingly) need to pad back to the original length
    components, seqQ, seqK = full_attribution.shape
    full_attribution_padded = t.zeros((components, seqK, seqK), device=full_attribution.device) #.fill_(float("-inf"))
    full_attribution_padded_scaled_positive = full_attribution_padded.clone()
    full_attribution_padded_scaled_negative = full_attribution_padded.clone()
    full_attribution_padded_scaled_positive[:, (seq_pos if len(seq_pos) > 1 else slice(None))] = full_attribution_scaled_positive
    full_attribution_padded_scaled_negative[:, (seq_pos if len(seq_pos) > 1 else slice(None))] = full_attribution_scaled_negative

    if mode == "small":
        html_pos = attention_patterns(
            attention = full_attribution_padded_scaled_positive,
            tokens = token_str,
            attention_head_names = labels,
        )
        html_neg = attention_patterns(
            attention = full_attribution_padded_scaled_negative,
            tokens = token_str,
            attention_head_names = labels,
        )
        html_dict = {"_pos": html_pos, "_neg": html_neg}
    else:
        html_all = attention_heads(
            attention = full_attribution_padded_scaled_positive - full_attribution_padded_scaled_negative,
            tokens = token_str,
            attention_head_names = labels,
        )
        html_dict = {"": html_all}
    
    # Get html as strings, in a list
    html_str = [str(html_dict[""])] if (mode != "small") else [f"<h1>Positive attribution</h1>{str(html_dict['_pos'])}", f"<h1>Negative attribution</h1>{str(html_dict['_neg'])}"]

    # Open in browser if required
    if return_mode == "browser":
        idx = 0
        filename_end = ["_pos", "_neg"] if mode == "small" else [""]
        while (Path.cwd() / f"attention{filename_end[0]}_{idx}.html").exists():
            idx += 1
        for end, html in zip(filename_end, html_str):
            file_path = Path.cwd() / f"attention{end}_{idx}.html"
            file_url = 'file://' + str(file_path.resolve())
            with open(str(file_path), "w") as f:
                f.write(html)
            result = webbrowser.open(file_url)
            if not result:
                raise RuntimeError(f"Failed to open {file_url} in browser. However, the file was saved to {file_path}, so you can download it and open it manually.")
    elif return_mode == "html":
        html_list = [HTML(_html_str) for _html_str in html_str]
        return html_list[0] if len(html_list) == 1 else html_list
    else:
        for _html_str in html_str:
            display(HTML(_html_str))
