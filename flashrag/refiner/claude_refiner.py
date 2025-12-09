from typing import List
import os
from openai import OpenAI
from tqdm import tqdm
from flashrag.refiner.refiner import BaseRefiner

class ClaudeRefiner(BaseRefiner):
    """Refiner using Claude models via CMU AI Gateway (OpenAI-compatible API)."""

    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AI_GATEWAY_API_KEY or OPENAI_API_KEY environment variable not set.")
        
        self.base_url = "https://ai-gateway.andrew.cmu.edu/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        # Default to sonnet if not specified, though config should usually specify it
        self.model_name = self.model_path if self.model_path else "claude-sonnet-4-20250514-v1:0"

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            # Try to extract title if possible, otherwise just use content
            lines = content.split("\n")
            if len(lines) > 1:
                title = lines[0]
                text = "\n".join(lines[1:])
            else:
                title = "Unknown"
                text = content
            format_reference += f"Doc {idx+1} (Title: {title}):\n{text}\n\n"
        return format_reference

    def batch_run(self, dataset, batch_size=None) -> List[str]:
        output = []
        for item in tqdm(dataset, desc="Refining with Claude: "):
            question = item.question
            retrieval_result = item.retrieval_result
            docs_text = self.format_reference(retrieval_result)

            # Construct prompt for Claude
            # We want Claude to act as a refiner/pruner.
            system_prompt = "You are a helpful assistant that refines retrieved documents for RAG. Your task is to select and extract the most relevant information from the provided documents to answer the user's question. Output only the refined information. If the documents contain no relevant information, output 'No relevant information found.'"
            
            user_prompt = f"Question: {question}\n\nRetrieved Documents:\n{docs_text}\n\nPlease refine the above documents to contain only information relevant to answering the question."

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0, # Deterministic output
                )
                refined_text = response.choices[0].message.content
                output.append(refined_text)
            except Exception as e:
                print(f"Error calling Claude API: {e}")
                # Fallback to original docs or empty string? Let's keep original docs for now or maybe just error message
                output.append(f"Error refining documents: {e}")

        return output
