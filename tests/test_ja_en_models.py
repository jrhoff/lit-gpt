# I separated this file for the time being for ease-of-development

import sys
from functools import partial
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch
from conftest import RunIf
from lightning.fabric.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_2_2

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.config as config_module

@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)

def test_against_hf_shisa_base(device, dtype):
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        "shisa-base-7b-v1",
        padded_vocab_size=120074,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
    )

    T = 5
    theirs_config = MistralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MistralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)

def test_against_hf_shisa_v1(device, dtype):
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        "shisa-base-7b-v1",
        padded_vocab_size=120074,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
    )
    T = 5
    theirs_config = MistralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MistralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)

def test_against_hf_elyza_instruct(device, dtype):
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        "shisa-base-7b-v1",
        padded_vocab_size=120074,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
    )
    T = 5
    theirs_config = MistralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MistralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)

@torch.inference_mode()
# @pytest.mark.parametrize(
#     "ours_kwargs",
#     [{"name": "Llama-2-7b-hf"}, {"name": "CodeLlama-7b-hf"}, {"name": "Llama-2-70b-chat-hf", "n_query_groups": 1}],
# )
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_hf_llama2(device, dtype):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        name="ELYZA-japanese-Llama-2-7b", padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, intermediate_size=86
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)