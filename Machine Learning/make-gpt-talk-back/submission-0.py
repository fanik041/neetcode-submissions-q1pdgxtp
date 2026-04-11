import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        output_text = ""

        for i in range(new_chars):

            # 1. Crop context
            context = context[:, -context_length:]

            # 2. Forward pass → get probabilities of last token
            logits = model(context)  # (B, T, vocab)
            last_logits = logits[:, -1, :]  # (B, vocab)
            probs = torch.softmax(last_logits, dim=-1)

            # Reset generator state BEFORE sampling (important for reproducibility)
            generator.set_state(initial_state)

            # 3. Sample next token
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)

            # 4. Append token to context
            context = torch.cat((context, next_token), dim=1)

            # 5. Convert token to character
            token_id = next_token.item()
            output_text += int_to_char[token_id]

        return output_text