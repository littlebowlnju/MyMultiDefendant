"""
    generate data for fact refiner by utilizing CAIL2018
    @auther: juju
"""
import argparse
from collections import defaultdict
import pickle
import random
import jsonlines
from enum import Enum, auto


# 拼接两段案件事实的方式
class SpliceMethod(Enum):
    # 顺序拼接
    SEQUENTIAL = auto()
    # 交叉拼接
    INTERLEAVED = auto()
    # 插入拼接，将case2插入到case1中
    INSERTED = auto()
    # 上下文嵌入拼接，在两端事实之间插入一段无关文本
    CONTEXTUAL = auto()

random_sentences = [
    "此处信息无关紧要，仅作为填充使用。", "这里插入无关描述，以提高模型鲁棒性。", "注意，这里的信息与案件无关，请继续阅读后续内容。", "天空蓝色草地绿色，书本上的字迹模糊不清。", "桌子上散落着各种无关的物品，比如铅笔、纸张。", "他在散步时随口说了些无关紧要的话，如天气变化等琐事。", "正如莎士比亚所言：全世界都是舞台。", "窗外的山川湖泊风景如画，鸟语花香，与此处讨论的主题无关。", "在一个遥远的地方，有一座静静的小村庄，孩子们在河边嬉戏。", "深秋的公园里，落叶缓缓飘落，一片宁静与和谐。", "如老话所说：天无绝人之路。", "在这之后，", "在此之前，", "另外，", "还有另外一起案件。"
]

# args
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cail_path', type=str, default='data/cail2018_big.json', help='path to the CAIL data file')
    parser.add_argument('--output_path', type=str, default='./data/merged_facts_from_cail.jsonl', help='path to the output file')
    # 对于每个罪名类型，生成的案件事实中被告人是同罪名的比例
    parser.add_argument('--same_charge_rate', type=float, default=0.5, help='rate of generating data with same charges')

    parser.add_argument('--spliced_data_num', type=int, default=20000, help='number of data to generate from cail')

    args = parser.parse_args()
    return args


# 利用大模型从多被告人案件中生成数据
def extract_refined_fact_from_multi_defendant_cases(data_path, output_path, num: int = 1000):
    # TODO
    pass


# 拼接两个案件，并返回最终的数据
def splice_two_cases(case1, case2, method: SpliceMethod = SpliceMethod.SEQUENTIAL, use_both: bool = False):
    # use_both: 随机概率，是仅用case1作为refined-fact，还是两个都各自作为refined-fact
    merged_fact = ''
    if method == SpliceMethod.SEQUENTIAL:
        merged_fact = case1["fact"] + case2["fact"]
    elif method == SpliceMethod.INTERLEAVED:
        sentences1 = case1["fact"].split('。')
        sentences1 = [s + '。' for s in sentences1 if s]
        sentences2 = case2["fact"].split('。')
        sentences2 = [s + '。' for s in sentences2 if s]
        i, j = 0, 0
        while i < len(sentences1) or j < len(sentences2):
            if i < len(sentences1):
                merged_fact += sentences1[i]
                i += 1
            if j < len(sentences2):
                merged_fact += sentences2[j]
                j += 1
    elif method == SpliceMethod.INSERTED:
        # 将case2插入到case1的某两个句子之间
        sentences1 = case1["fact"].split('。')
        sentences1 = [s + '。' for s in sentences1 if s]
        if len(sentences1) == 1:
            merged_fact = case2["fact"] + case1["fact"]
        else:
            inserted_idx = random.randint(1, len(sentences1) - 1)
            merged_fact = ''.join(sentences1[:inserted_idx]) + case2["fact"] + ''.join(sentences1[inserted_idx:])
    elif method == SpliceMethod.CONTEXTUAL:
        merged_fact = case1["fact"] + random.choice(random_sentences) + case2["fact"]
    result_obj1 = {
        "merged_fact": merged_fact,
        "refined_fact": case1["fact"],
        "meta": case1["meta"]
    }
    result_obj2 = None
    if use_both:
        result_obj2 = {
            "merged_fact": merged_fact,
            "refined_fact": case2["fact"],
            "meta": case2["meta"]
        }
    return result_obj1, result_obj2


# 读取数据并按照罪名类型分类存储
def sort_cail_by_charges(cail_path):
    charge_cases = defaultdict(list)
    with jsonlines.open(cail_path) as reader:
        for case in reader:
            charge_cases[case["meta"]["accusation"][0]].append(case)
    print(charge_cases.keys())
    # save
    with open('./resources/charge_cases.pkl', 'wb') as f:
        pickle.dump(charge_cases, f)


# 从CAIL2018数据中生成数据
# 生成的数据包含：与同罪案件事实拼接、与不同罪案件事实拼接、不拼接
def generate_data_from_cail(output_path, same_charge_rate, num: int = 20000):
    with open('./resources/charge_cases.pkl', 'rb') as f:
        charge_cases = pickle.load(f)
    charges = list(charge_cases.keys())
    charge_case_cnt = {k: len(v) for k, v in charge_cases.items()}
    total_case_cnt = sum(charge_case_cnt.values())
    print(charge_case_cnt)
    # 按照罪名类型数量生成数据
    with jsonlines.open(output_path, 'a') as writer:
        for charge in charges:
            needed_cnt = int(num * charge_case_cnt[charge] / total_case_cnt)
            if needed_cnt == 0:
                continue
            cnt = 0
            while cnt < needed_cnt:
                case_1 = random.choice(charge_cases[charge])
                with_same_charge = random.random() < same_charge_rate
                if with_same_charge:
                    case_2 = random.choice(charge_cases[charge])
                else:
                    case_2 = random.choice(charge_cases[random.choice(charges)])
                concate_method = random.choice(list(SpliceMethod))
                # 是否同时利用两个案件作为refined-fact
                use_both = random.random() < 0.4
                result_obj1, result_obj2 = splice_two_cases(case_1, case_2, concate_method, use_both)
                writer.write(result_obj1)
                cnt += 1
                if result_obj2 is not None and cnt < needed_cnt:
                    writer.write(result_obj2)
                    cnt += 1



if __name__ == "__main__":
    # 生成罪名-案件数据并存储
    # cail_path = ''
    # sort_cail_by_charges(cail_path)

    args = parse()
    generate_data_from_cail(args.output_path, args.same_charge_rate, args.spliced_data_num)