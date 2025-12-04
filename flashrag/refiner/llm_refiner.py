# flashrag/refiner/llm_refiner.py

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashrag.refiner import BaseRefiner

DEFAULT_REFINER_PROMPT = """You are a context compressor for open-domain QA.
Given a question and several retrieved passages, rewrite them into a shorter,
well-structured context that preserves all information needed to answer the question.
Do NOT answer the question. Only output the compressed context.

Question: {question}

Retrieved passages:
{passages}

Compressed context:
"""


class LLMAbstractiveRefiner(BaseRefiner):
    """Summarize retrieved passages with a decoder-only LLM (e.g., Llama-3, Mistral)."""

    def __init__(self, config):
        super().__init__(config)
        self.max_input_len = config["refiner_max_input_length"] or 2048
        self.max_output_len = config["refiner_max_output_length"] or 512
        self.batch_size = config["refiner_batch_size"] if "refiner_batch_size" in config else 1
        self.batch_size = self.batch_size or 1
        self.prompt_template = config["refiner_prompt"] if "refiner_prompt" in config else DEFAULT_REFINER_PROMPT
        if self.prompt_template is None:
            self.prompt_template = DEFAULT_REFINER_PROMPT

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"

        # place the refiner on a specific device to avoid competing with vLLM
        use_cuda = "cuda" in str(self.device) and torch.cuda.is_available()
        dtype_name = (
            config["refiner_torch_dtype"]
            if "refiner_torch_dtype" in config
            else ("float16" if use_cuda else "float32")
        )
        dtype = getattr(torch, dtype_name)
        model_kwargs = {"torch_dtype": dtype}

        if "refiner_device_map" in config:
            model_kwargs["device_map"] = config["refiner_device_map"]
        elif use_cuda:
            if "cuda:" in str(self.device):
                model_kwargs["device_map"] = {"": str(self.device)}
            else:
                # if only one GPU is visible, pin to that ID to avoid accidental split with vLLM
                visible_cuda = torch.cuda.device_count()
                target = "cuda:0" if visible_cuda >= 1 else "cuda"
                model_kwargs["device_map"] = {"": target}
        else:
            model_kwargs["device_map"] = {"": "cpu"}

        if "refiner_max_memory" in config:
            max_mem = config["refiner_max_memory"]
            # normalize keys like "cuda" -> "cuda:0" for HF/accelerate compatibility
            if isinstance(max_mem, dict):
                normalized = {}
                for k, v in max_mem.items():
                    if k == "cuda":
                        normalized["cuda:0"] = v
                    else:
                        normalized[k] = v
                max_mem = normalized
            model_kwargs["max_memory"] = max_mem

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        if not use_cuda:
            self.model.to(self.device)
        self.model.eval()
        self.model_device = getattr(self.model, "device", self.device)

    def format_reference(self, retrieval_result):
        formatted_reference = []
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"] or doc_item["text"] or ""
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:]) if "\n" in content else content
            formatted_reference.append(f"Doc {idx+1}(Title: {title}) {text}")
        return "\n".join(formatted_reference).strip()

    def _build_prompt(self, question, retrieval_result):
        formatted_reference = self.format_reference(retrieval_result)
        return self.prompt_template.format(question=question, passages=formatted_reference)

    def batch_run(self, dataset):
        questions = dataset.question
        retrieval_results = dataset.retrieval_result

        outputs = []
        for start in tqdm(range(0, len(questions), self.batch_size), desc="Refining process: "):
            batch_questions = questions[start : start + self.batch_size]
            batch_retrieval = retrieval_results[start : start + self.batch_size]
            prompts = [self._build_prompt(q, docs) for q, docs in zip(batch_questions, batch_retrieval)]

            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_input_len,
                return_tensors="pt",
            ).to(self.model_device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_output_len,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for seq, attn in zip(generated, inputs["attention_mask"]):
                prompt_len = int(attn.sum())
                new_tokens = seq[prompt_len:]
                outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

            torch.cuda.empty_cache()

        return outputs
