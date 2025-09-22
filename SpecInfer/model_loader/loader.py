import logging

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # class for better type annotations
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate
logger = logging.getLogger(__name__)

class ModelLoader:
    @classmethod
    def load_model(
        cls,
        model_name_or_path: str,
        tp_size: int = 1,
    ) -> "_BaseModelWithGenerate":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            tp_size=tp_size,
            tp_plan="auto"
        )
        return model

    @classmethod
    def load_model_and_tokenizer(
        cls,
        model_name_or_path: str,
        tp_size: int = 1,
    ) -> tuple["_BaseModelWithGenerate", transformers.PreTrainedTokenizerBase]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            tp_size=tp_size,
            tp_plan="auto"
        )
        return model, tokenizer
    
    @classmethod
    def _load_model_with_parallel(
        cls,
        model_name_or_path: str,
        tp_size: int,
    ) -> "_BaseModelWithGenerate":
        raise NotImplementedError(
            "This function is useless now since transformers have supported tensor parallel."
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
