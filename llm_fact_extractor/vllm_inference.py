from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
import re
import json
import torch

model_name = "/home/hwh/hf_models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=0.95, max_tokens=512)

torch.cuda.empty_cache()

llm = LLM(model=model_name, dtype="bfloat16", device=torch.device("cuda:0"))

def generate_message_extract_fact(fact_description, defendant_name):
    prompt = f"""请从下述案件事实描述中提取出与被告人{defendant_name}直接相关或间接相关的事实。将提取结果拼接后直接输出，不添加额外描述。
###案件事实：{fact_description}"""

    messages = [
        {"role": "system", "content": "你是一个经验丰富的法官。请分析包含多个被告人的复杂案件事实，提取与各被告人相关的案件事实，排除无关信息。"},
        {"role": "user", "content": prompt}
    ]
    return messages

def generate_message_extract_joint_crime_case(fact_description, defendant_names):
    prompt = '请判断下述包含多个被告人的案件事实中，是否存在共同犯罪的情况，以及各被告人的犯罪角色（主犯/从犯/胁从犯）。请按照JSON格式输出结果：如，{"共同犯罪": true, "主犯": ["张三"], "从犯": ["李四", "王五"], "胁从犯": null} 或 {"共同犯罪": false}。'
    prompt += f"""
    ###案件事实: {fact_description}
    ###被告人列表：{defendant_names}
    """
    messages = [
        {"role": "system", "content": "你是一个经验丰富的法官，理解共同犯罪是指二人以上共同故意犯罪。请分析以下多被告人案件中的共同犯罪情况。"},
        {"role": "user", "content": prompt}
    ]
    return messages


# batch inference to extract defendant-related facts
def read_file_in_batches(file_path, batch_size=8):
    with jsonlines.open(file_path) as reader:
        batch_case_info, batch_messages = [], []
        cur_batch_size = 0
        for i, case in enumerate(reader):
            outcomes = case["outcomes"]
            for outcome in outcomes:
                if cur_batch_size == batch_size:
                    yield batch_case_info, batch_messages
                    batch_case_info = []
                    batch_messages = []
                    cur_batch_size = 0
                batch_case_info.append({"case_id": case["case_id"], "defendant": outcome["name"], "judgment": outcome["judgment"]})
                batch_messages.append(generate_message_extract_fact(case["fact"], outcome["name"]))
                cur_batch_size += 1
        if batch_messages:
            yield batch_case_info, batch_messages

def inference_extract_facts(file_path, output_file):
    with jsonlines.open(output_file, 'a') as writer:
        for batch_case_info, batch_messages in tqdm(read_file_in_batches(file_path)):
            texts = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate(texts, sampling_params)
            for case_info, output in zip(batch_case_info, outputs):
                # prompt = output.prompt
                generated_text = output.outputs[0].text
                # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                writer.write({"case_id": case_info["case_id"], "defendant": case_info["defendant"], "judgment": case_info["judgment"], "extracted_fact": generated_text})

def get_json_obj_from_generated_text(text):
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        json_obj = match.group(1)
        print(json_obj)
        try:
            return json.loads(json_obj)
        except json.JSONDecodeError:
            print("JSONDecodeError. skipping...")
            return None
    return None

def inference_extract_join_crime_cases(file_path, output_file):
    log_file = "/home/hwh/my_multi_defendant/llm_fact_extractor/join_crime_infer_result.log"
    with open(log_file, "a") as log:
        with jsonlines.open(output_file, "a") as writer:
            with jsonlines.open(file_path, "r") as reader:
                for i, case in enumerate(reader):
                    if i < 1677:
                        continue
                    text = tokenizer.apply_chat_template(generate_message_extract_joint_crime_case(case["fact"] + case["court_view"], case["defendants"]), tokenize=False, add_generation_prompt=True)
                    outputs = llm.generate([text], sampling_params)
                    for output in outputs:
                        generated_text = output.outputs[0].text
                        log.write(f"case {case['case_id']} Generated text: {generated_text}\n")
                        json_obj = get_json_obj_from_generated_text(generated_text)
                        if json_obj and json_obj["共同犯罪"]:
                            write_obj = case
                            write_obj["joint_crime_meta"] = json_obj
                            writer.write(write_obj)


if __name__ == "__main__":
    inference_extract_facts("/home/hwh/my_multi_defendant/data/cmdl/valid_smaller_with_id.jsonl", "/home/hwh/my_multi_defendant/data/curated/valid_smaller_extracted_fact.jsonl")
    # inference_extract_join_crime_cases("/home/hwh/my_multi_defendant/data/cmdl/train_smaller_with_id.jsonl", "/home/hwh/my_multi_defendant/data/extracted_fact/train_smaller_joint_crime_cases.jsonl")