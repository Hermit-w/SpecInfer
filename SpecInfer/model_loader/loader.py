import logging
import os
from typing import TYPE_CHECKING

import transformers
from deprecated.sphinx import deprecated
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    # class for better type annotations
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

logger = logging.getLogger(__name__)
rank = int(os.environ.get("RANK", 0))


class ModelLoader:
    @classmethod
    def load_model(
        cls,
        model_name_or_path: str,
        tp_size: int = 1,
        *,
        torch_dtype: str = "auto",
    ) -> "_BaseModelWithGenerate":
        config = AutoConfig.from_pretrained(model_name_or_path)
        # Direct load will lead to error
        # Referring https://github.com/huggingface/transformers/issues/41092
        if hasattr(config, "num_key_value_heads"):
            if not hasattr(config, "base_model_tp_plan"):
                raise ValueError(f"Existing model: {config.architectures[0]} doesn't support tp") # noqa
            if config.num_key_value_heads % tp_size != 0:
                logger.warning(f"Existing model has kv_heads={config.num_key_value_heads} while setting tp_size={tp_size}, which may lead to error") # noqa
                if tp_size % config.num_key_value_heads != 0:
                    logger.error(f"Existing method doesn't support {config.architectures[0]} with tp_size={tp_size}")
                    raise ValueError(f"Existing method doesn't support {config.architectures[0]} with tp_size={tp_size}") # noqa
                logger.warning("Try to pass our customized tp_plan to model")
                plan = config.base_model_tp_plan
                # This is solution provided by core maintainer of transformers
                # Referring https://github.com/huggingface/transformers/issues/40953#issuecomment-3311635988
                plan["layers.*.self_attn.q_proj"] = "colwise_rep"
                plan["layers.*.self_attn.k_proj"] = "colwise_rep"
                plan["layers.*.self_attn.v_proj"] = "colwise_rep"
                plan["layers.*.self_attn.o_proj"] = "rowwise_rep"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            tp_size=tp_size,
            tp_plan="auto",
            dtype=torch_dtype,
        )
        return model

    @classmethod
    def load_model_and_tokenizer(
        cls,
        model_name_or_path: str,
        tp_size: int = 1,
        *,
        torch_dtype: str = "auto",
    ) -> tuple["_BaseModelWithGenerate", transformers.PreTrainedTokenizerBase]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = cls.load_model(
            model_name_or_path,
            tp_size,
            torch_dtype=torch_dtype,
        )
        return model, tokenizer

    @deprecated("transformers have supported tensor parallel")
    @classmethod
    def _load_model_with_parallel(
        cls,
        model_name_or_path: str,
        tp_size: int,
        *,
        torch_dtype: str = "auto",
    ) -> "_BaseModelWithGenerate":
        raise NotImplementedError(
            "This function is deprecated now since transformers have supported tensor parallel."
        )
        # logger.info(f"Loading model from {model_name_or_path} with tensor parallelism of size {tp_size}.")
        # config = AutoConfig.from_pretrained(model_name_or_path)
        # if not check_support_model(config):
        #     raise NotImplementedError("Current model architecture has not been supported yet.")
        # if not check_tensor_parallel_legality(config, tp_size):
        #     raise ValueError("Current tp size is not compatiable with the model architecture.")
        # logger.debug("Loading empty model with no memory usage.")
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config)
        # device_mesh = DeviceMesh("cuda", torch.arange(tp_size, dtype=torch.int))
        # tp_plan = get_tensor_parallel_plan(model_name_or_path, config)
        # module = parallelize_module(model, device_mesh, tp_plan)
        # logger.debug("Loadig weight with allocated memory.")
        # model = load_checkpoint_and_dispatch(
        #     module,
        #     checkpoint=model_name_or_path
        # )
        # return model
