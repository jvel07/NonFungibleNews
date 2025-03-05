import os
# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import pipeline
import torch


class LLMSummarizer:
    def __init__(
            self,
            model_name = "meta-llama/Llama-3.1-8B-Instruct",
            # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            system_prompt = "You are a helpful assistant.",
        ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        # self.rewrite_prompt = rewrite_prompt
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        print(f"{self.model_name} initialized")

    def cleanup(self):
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            self.pipeline.model.to("cpu")
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()

    def rewrite_if_exceeds(self, text: str,  max_new_tokens: int, rewrite_prompt: str) -> str:
        messages = [
            {
             "role": "system",
             "content": rewrite_prompt
            },
            {"role": "user", "content": text},
        ]
        output = self.pipeline(messages, max_new_tokens=max_new_tokens)
        shortened_text = output[0]["generated_text"][-1]['content']
        return shortened_text

    def summarize_text(self, text: str, max_new_tokens: int, rewrite_prompt: str) -> str:
        messages = [
            {"role": "system",
             "content": self.system_prompt
             },
            {"role": "user", "content": text},
        ]
        output = self.pipeline(messages, max_new_tokens=max_new_tokens)
        summarized_styled = output[0]["generated_text"][-1]['content']

        # if len(summarized_styled) > 280:
        #     print("/n shortened!/n")
        #     summarized_styled = self.rewrite_if_exceeds(summarized_styled, 280, rewrite_prompt)

        print("\n original text:", text)
        print("\n summarized text:", summarized_styled)
        print(f"{'=' * 90}\n")
        return summarized_styled

if __name__ == "__main__":
    text = "Give the python code to load a dataset from a csv file."
    model = LLMSummarizer()
    reply = model.summarize_text(text, 280, "You are a helpful assistant.")
    print(reply)