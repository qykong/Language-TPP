import torch
from modelscope import AutoModelForCausalLM, AutoProcessor
import numpy as np


sample_sequence = """<|im_start|>system
textual representation of an event sequence denoted by event times in float Byte-tokens (each number as 4 byte tokens) along with textual event types
INFO: This sequence is a product review event from an Amazon user where event type is product category
<|im_end|>
<|im_start|>sequence
<|start_of_event|><|type_prefix|>Children Clothing<|description_prefix|>Two Stars<|time_prefix|>0<|end_of_event|><|start_of_event|>"""

# Special tokens
SPECIAL_TOKENS = {
    "START_OF_EVENT": "<|start_of_event|>",
    "END_OF_EVENT": "<|end_of_event|>",
    "TIME_PREFIX": "<|time_prefix|>",
    "DESCRIPTION_PREFIX": "<|description_prefix|>",
    "TYPE_PREFIX": "<|type_prefix|>",
    "DESCRIPTION_GENERATION": "<|description_generation|>",
    "TYPE_PREDICTION": "<|type_prediction|>",
    "TIME_PREDICTION": "<|time_prediction|>",
}


def float32_to_bytes_big_endian(value):
    bytes_obj = np.array([value], dtype=np.float32).tobytes()
    return bytes_obj[3], bytes_obj[2], bytes_obj[1], bytes_obj[0]


def float32_to_byte_tokens(value):
    int1, int2, int3, int4 = float32_to_bytes_big_endian(value)
    return (
        f"<|byte_{int1}|>",
        f"<|byte_{int2}|>",
        f"<|byte_{int3}|>",
        f"<|byte_{int4}|>",
    )


def bytes_to_float32_big_endian(first_byte, second_byte, third_byte, fourth_byte):
    # Create bytes object in big-endian order
    bytes_obj = bytes([fourth_byte, third_byte, second_byte, first_byte])
    return np.frombuffer(bytes_obj, dtype=np.float32)[0]


def byte_tokens_to_float32(token_list):
    if isinstance(token_list, list):
        token_list = "".join(token_list)

    token_ints = [int(bs.split("|>")[0]) for bs in token_list.split("byte_")[1:]]
    return bytes_to_float32_big_endian(*token_ints)


class LanguageTPPModel:
    def __init__(self, model_path: str = "./Language_TPP_0___5B"):
        self.tokenizer = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

        self._add_special_tokens()

    def predict_next_event(self, event_sequence: str):
        self._convert_to_byte_tokens(event_sequence)

        task_outputs = []
        for task_token in [
            "<|type_prediction|>",
            "<|time_prediction|>",
            "<|description_generation|>",
        ]:
            inputs = self.tokenizer(
                event_sequence + task_token, return_tensors="pt", padding=True
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                stop_strings=["<|end_of_event|>"],
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            decoded_output = self.tokenizer.decode(
                outputs[0], skip_special_tokens=False
            )

            task_outputs.append(decoded_output.split(task_token)[1])

        return task_outputs

    def _convert_to_byte_tokens(self, event_sequence: str):
        events = event_sequence.split("<|time_prefix|>")
        converted_event_sequence = events[0]
        for event in events[1:]:
            byte_tokens = float32_to_byte_tokens(
                float(event.split("<|end_of_event|>")[0])
            )
            converted_event_sequence += (
                "<|time_prefix|>" + "".join(byte_tokens) + "<|end_of_event|>"
            )
        return converted_event_sequence
