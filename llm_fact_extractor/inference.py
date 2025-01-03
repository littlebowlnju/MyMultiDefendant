import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from tqdm import tqdm

model_name = "/home/hwh/hf_models/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("using device: ", model.device)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# 抽取与某个被告人相关的案件事实
def single_inference_test():
    defendant_name = "丁永生"
    original_fact = "淮安市淮安区人民检察院指控：2019年5月30日17时11分左右，被害人徐某乙驾驶电动自行车沿G233国道由南向北行驶至苏北灌溉总渠大桥上时，操作不当、未安全驾驶，撞上前方同向停在路东侧被告人童兆兵驾驶的电动三轮车的左后尾部后摔倒，倒到由南向北被告人丁永生驾驶的无号牌变型拖拉机右后轮处，致徐某乙及电动自行车被变型拖拉机碾轧，造成车辆损坏，徐某乙受伤的交通事故。事故发生后，丁永生和童兆兵均驾车逃逸。徐某乙经淮安市淮安区人民医院抢救无效于6月15日11时05分死亡。经淮安市公安局淮安分局交通警察大队认定，丁永生、童兆兵负事故全部责任，徐某乙不负事故责任。另查明：被告人童兆兵在案发后主动投案，如实供述自己的犯罪事实；被告人丁永生归案后如实供述自己的犯罪事实。二人均自愿认罪认罚。2019年8月27日，被告人丁永生与被害人近亲属达成赔偿协议，被告人丁永生赔偿被害方经济损失39.5万元，并取得谅解。2019年9月2日，被告人童兆兵与被害人近亲属达成了赔偿协议，被告人童兆兵赔偿了被害方经济损失36万元，并取得了谅解。还查明：淮安市公安局淮安分局交通警察大队出具《道路交通事故认定书》事故形成原因认定丁永生、童兆兵负事故全部责任，是基于《中华人民共和国道路交通事故安全法实施条例》第九十二条的规定，&ldquo;发生道路交通事故后当事人逃逸的，逃逸的当事人承担全部责任&rdquo;。"

    prompt = f"""请从下述案件事实描述中提取出与被告人{defendant_name}直接相关或间接相关的事实。将提取结果拼接后直接输出，不需要进行其他处理。
    ###案件事实：{original_fact}"""

    messages = [
        {"role": "system", "content": "你是一个经验丰富的法官。请分析包含多个被告人的复杂案件事实，提取与各被告人相关的案件事实，排除无关信息。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def generate_message_extract_fact(fact_description, defendant_name):
    prompt = f"""请从下述案件事实描述中提取出与被告人{defendant_name}直接相关或间接相关的事实。将提取结果拼接后直接输出，不添加额外描述。
###案件事实：{fact_description}"""

    messages = [
        {"role": "system", "content": "你是一个经验丰富的法官。请分析包含多个被告人的复杂案件事实，提取与各被告人相关的案件事实，排除无关信息。"},
        {"role": "user", "content": prompt}
    ]
    return messages


def generate_message_extract_joint_crime_case(fact_description, defendant_names):
    prompt = '请判断下述包含多个被告人的案件事实中，是否存在共同犯罪的情况，以及各被告人的犯罪角色（主犯/从犯/胁从犯）。请按照JSON格式输出结果：如，{"共同犯罪": true, "主犯": "张三", "从犯": ["李四", "王五"], "胁从犯": null} 或 {"共同犯罪": false}。'
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
def read_file_in_batches(file_path, batch_size=3):
    with jsonlines.open(file_path) as reader:
        batch_case_info, batch_messages = [], []
        cur_batch_size = 0
        for i, case in enumerate(reader):
            if i < 438:
                continue
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


def extract_defendant_related_facts(file_path, output_file):
    with jsonlines.open(output_file, 'a') as writer:
        for batch_case_info, batch_messages in tqdm(read_file_in_batches(file_path)):
            # print(batch_messages)
            texts = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            generated_ids_batch = model.generate(**model_inputs, max_new_tokens=512)
            generated_ids_batch = generated_ids_batch[:, model_inputs.input_ids.shape[1]:]
            responses = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
            for case_info, response in zip(batch_case_info, responses):
                writer.write({"case_id": case_info["case_id"], "defendant": case_info["defendant"], "judgment": case_info["judgment"], "extracted_fact": response})


# 提取被告人是否为共同犯罪中的主犯/从犯/不适用/共同犯罪、无主从之分
def single_inference_test_join_crime():
    fact_description = "某地人民检察院指控：被告人段某某、管某某于2016年12月31日17时许，到巧家县xx乡xx村xx社失主杨某某家，管某某趁杨某某不备将杨某某放于卧室一包内的一张银行卡窃取交给段某某，随后段某某到银行以提取现金和转账的方式将杨某某银行卡内的50000元人民币盗走，当晚段某某将银行卡交管某某放回原处。案发后，二被告人已退还失主被盗的现金。现管某某怀孕。上述事实，被告人段某某、管某某在开庭审理过程中均无异议，并有报案笔录，抓获经过，刑事判决书，释放证明，收条，失主杨某某的陈述，证人肖某某的证言，辨认笔录及照片，户口证明等证据证实，足以认定。本院认为，盗窃罪是指盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的行为。被告人段某某、管某某盗窃人民币50000元，属数额巨大，符合盗窃罪的主客观构成要件，构成盗窃罪。公诉机关指控的罪名成立，应依法处罚。被告人段某某、管某某到案后，如实供述犯罪事实，认罪态度和悔罪表现较好，可以从轻处罚。在共同犯罪中，被告人段某某、管某某作用相当，不宜区分主从。被告人段某某在刑罚执行完毕以后，在五年以内再犯应当判处有期徒刑以上刑罚之罪的，是累犯，应当从重处罚。被盗窃的现金已返还失主，未给失主造成损失，可以酌情从轻处罚。为体现“宽严相济”的刑事政策，综合全案考虑，对被告人段某某从轻处罚；对被告人管某某从轻处罚并适用缓刑"
    defendants = ["段某某", "管某某"]
    prompt = generate_message_extract_joint_crime_case(fact_description, defendants)
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def extract_joint_crime_case(file_path, output_file):
    with jsonlines.open(output_file, 'a') as writer:
        with jsonlines.open(file_path) as reader:
            for case in reader:
                pass
                
if __name__ == "__main__":
    # extract_defendant_related_facts("/home/hwh/my_multi_defendant/data/cmdl/train_smaller_with_id.jsonl", "/home/hwh/my_multi_defendant/data/extracted_fact/train_smaller_extracted_fact.jsonl")
    single_inference_test_join_crime()