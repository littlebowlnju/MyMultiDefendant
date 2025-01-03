# Author: jujuuhuang
# Date: 2024-12-25
# Description: additional processing of the dataset
import jsonlines
import json
from collections import defaultdict
from common import add_to_log, merge_two_jl_files

top_30_charges = ['盗窃罪', '寻衅滋事罪', '开设赌场罪', '诈骗罪', '故意伤害罪', '走私、贩卖、运输、制造毒品罪', '非法拘禁罪', '聚众斗殴罪', '赌博罪', '敲诈勒索罪', '非法经营罪', '掩饰、隐瞒犯罪所得、犯罪所得收益罪', '抢劫罪', '妨害公务罪', '非法吸收公众存款罪', '故意毁坏财物罪', '容留他人吸毒罪', '滥伐林木罪', '组织、领导传销活动罪', '引诱、容留、介绍卖淫罪', '伪造、变造、买卖国家机关公文、证件、印章罪', '帮助信息网络犯罪活动罪', '贪污罪', '非法捕捞水产品罪', '污染环境罪', '职务侵占罪', '合同诈骗罪', '非法采矿罪', '假冒注册商标罪', '强迫交易罪']

penalty_type_rank = {
    "death_penalty": 5,
    "life_imprisonment": 4,
    "imprisonment": 3,
    "detention": 2,
    "surveillance": 1,
    "exemption": 0  # 免予刑事处罚
}


def get_top_30_charges():
    # 除开前30名外，其余的均视为rare，可以增加样本数量
    with open("/home/hwh/dataset/multi_defendants/log/charge_count_log.json", "r") as file:
        charge_count = json.load(file)
        idx = 0
        for charge, count in charge_count.items():
            if idx >= 30:
                break
            top_30_charges.append(charge)
            idx += 1
    print(top_30_charges)

def add_case_id(file, saved_file):
    with jsonlines.open(file) as reader:
        with jsonlines.open(saved_file, 'w') as writer:
            for i, obj in enumerate(reader):
                obj['case_id'] = i
                writer.write(obj)

def count_joint_crime_cases(file="/home/hwh/my_multi_defendant/data/curated/train_smaller_defendant_pairs.jsonl"):
    paired_nums = 0
    joint_crime_cases = set()
    with jsonlines.open(file, "r") as reader:
        for case in reader:
            if case["relation"] == "lead":
                paired_nums += 1
                joint_crime_cases.add(case["case_id"])
    add_to_log("/home/hwh/my_multi_defendant/logs/data_logs/generate_pairs_data.log", f"成功配对为lead关系的pair数量：{paired_nums}")
    with open("/home/hwh/my_multi_defendant/logs/data_logs/joint_crime_cases.jsonl", "w") as f:
        json.dump(list(joint_crime_cases), f)


def generate_non_joint_crimes_two_defendant_sample_pairs(original_file, saved_file):
    """ 将原始数据中两两被告人的数据组合成一对数据，其中包含·共同犯罪中主从犯·/·无相对关系·两种组合类型
    保存结果中每个pair包含：{"case_id": int, "original_fact": str, "defendant1": str, "defendant2": str, "defendant1_fact": str, "defendant2_fact": str, "defendant1_outcome": json, "defendant2_outcome": json, "relation": str}
    （defendant1_fact和defendant2_fact后续从另一个文件中抽取后加入）
    其中，relation对应三种类型: [1主2从: "lead"(限定主从犯关系中1一定是主，2是从); ?(这一项在模型中没有作用，视作none构建?)1主2主/1从2从: "equal"(共同犯罪中二者的地位相当); 非共同犯罪关系: "none"(无相对关系)]，此处lead关系需要是主犯的刑期比从犯的刑期长的才符合

    Args:
        original_file (str): 原始数据文件（非抽取单人事实、非共同犯罪）路径
        saved_file (str): 保存结果文件路径
        case_id_range (list): 访问的案件数据范围，包含左右边界
    """
    joint_crimes = set()
    with open("/home/hwh/my_multi_defendant/logs/data_logs/joint_crime_cases.jsonl") as f:
        joint_crimes = set(json.load(f))
    print("共同犯罪案件数量：", len(joint_crimes))
    with jsonlines.open(saved_file, "a") as writer:
        with jsonlines.open(original_file, "r") as reader:
            for i, case in enumerate(reader):
                if i in joint_crimes:
                    continue
                # 非共同犯罪，直接两两组合，并保证每个被告人出现2次 (前提是非共同犯罪是少数情况？)
                # 不可以让defendant2为空，在后续模型训练中会出现问题！！！
                # for defendant in case["outcomes"]:
                #     pair_json = {"case_id": case["case_id"], "original_fact": case["fact"], "defendant1": defendant["name"], "defendant2": None, "defendant1_fact": "", "defendant2_fact": "", "defendant1_outcome": defendant["judgment"], "defendant2_outcome": None, "relation": "none"}
                #     # 如果该被告人的罪名均属于top_30_charges，不再增加其出现次数，即设置defendant2为None
                #     for ac in defendant["judgment"]:
                #         if ac["accusation"] not in top_30_charges:
                #             pair_json["defendant2"] = defendant["name"]
                #             pair_json["defendant2_outcome"] = defendant["judgment"]
                #             break
                #     writer.write(pair_json)
                randomly_pair_unrelated_defendants(case, case["outcomes"], writer)


def penalty_type_and_length(penalty_obj):
    if penalty_obj["death_penalty"]:
        return "death_penalty", 0
    elif penalty_obj["life_imprisonment"]:
        return "life_imprisonment", 0
    elif penalty_obj["imprisonment"] > 0:
        return "imprisonment", penalty_obj["imprisonment"]
    elif penalty_obj["detention"] > 0:
        return "detention", penalty_obj["detention"]
    elif penalty_obj["surveillance"] > 0:
        return "surveillance", penalty_obj["surveillance"]
    else:
        return "exemption", 0


def check_penalty_term_relation(principal_term, accomplice_term):
    # 检查主从犯的刑期关系是否符合要求
    # death_penalty > life_imprisonment > imprisonment > detention > surveillance
    # 同个刑罚类型比较长度
    pricipal_term_type, principal_term_length = penalty_type_and_length(principal_term)
    accomplice_term_type, accomplice_term_length = penalty_type_and_length(accomplice_term)
    if penalty_type_rank[pricipal_term_type] > penalty_type_rank[accomplice_term_type]:
        return True
    elif penalty_type_rank[pricipal_term_type] == penalty_type_rank[accomplice_term_type]:
        return principal_term_length > accomplice_term_length
    else:
        return False


def check_principal_accomplice_relation(principal_outcome, accomplice_outcome):
    # 检查主从犯关系是否符合要求
    # 主从犯都犯相同的罪名，且主犯的刑期比从犯的刑期长
    principal_charges = set([ac["standard_accusation"] for ac in principal_outcome])
    accomplice_charges = set([ac["standard_accusation"] for ac in accomplice_outcome])
    if principal_charges != accomplice_charges:
        return False
    # final_penalty字段只针对多罪名被告人适用
    principal_penalty = principal_outcome["final_penalty"] if "final_penalty" in principal_outcome else principal_outcome[0]["penalty"]
    accomplice_penalty = accomplice_outcome["final_penalty"] if "final_penalty" in accomplice_outcome else accomplice_outcome[0]["penalty"]
    return check_penalty_term_relation(principal_penalty, accomplice_penalty)


def randomly_pair_unrelated_defendants(case_obj, defendant_info, writer):
    # 将案件中无主从关系无法配对的被告人两两组合写入saved_file
    # 两两组合，如果是奇数，最后一位被告人重复一次
    for i in range(0, len(defendant_info), 2):
        defendant1 = defendant_info[i]
        defendant2 = defendant_info[i + 1] if i + 1 < len(defendant_info) else defendant_info[i]
        paired_json = {"case_id": case_obj["case_id"], "original_fact": case_obj["fact"], "defendant1": defendant1["name"], "defendant2": defendant2["name"], "defendant1_fact": "", "defendant2_fact": "", "defendant1_outcome": defendant1["judgment"], "defendant2_outcome": defendant2["judgment"], "relation": "none"}
        writer.write(paired_json)


def generate_joint_crime_pairs(joint_crime_file, saved_file, lower_bound=None, upper_bound=None):
    satisfactory_joint_crime_cases = set()
    # 记录为主从犯关系的pair数量，最后计算占比？
    lead_pairs_num = 0
    with jsonlines.open(saved_file, "a") as writer:
        with jsonlines.open(joint_crime_file, "r") as reader:
            for obj in reader:
                if lower_bound and obj["case_id"] < lower_bound:
                    continue
                if upper_bound and obj["case_id"] >= upper_bound:
                    break
                if "共同犯罪" in obj["joint_crime_meta"] and obj["joint_crime_meta"]["共同犯罪"]:
                    available = True
                    if ("主犯" not in obj["joint_crime_meta"] or obj["joint_crime_meta"]["主犯"] is None) \
                        or (("从犯" not in obj["joint_crime_meta"] or ("从犯" in obj["joint_crime_meta"] and obj["joint_crime_meta"]["从犯"] is None)) \
                        and ("胁从犯" not in obj["joint_crime_meta"] or ("胁从犯" in obj["joint_crime_meta"] and obj["joint_crime_meta"]["胁从犯"] is None))):
                        add_to_log("/home/hwh/my_multi_defendant/logs/data_logs/generate_pairs_data.log", f"case_id: {obj['case_id']} 共同犯罪但不存在主从犯关系。")
                        continue
                    principal_names = set(obj["joint_crime_meta"]["主犯"])
                    accomplice_names = obj["joint_crime_meta"]["从犯"] if "从犯" in obj["joint_crime_meta"] and obj["joint_crime_meta"]["从犯"] is not None else []
                    accomplice_names.extend(obj["joint_crime_meta"]["胁从犯"] if "胁从犯" in obj["joint_crime_meta"] and obj["joint_crime_meta"]["胁从犯"] is not None else [])
                    accomplice_names = set(accomplice_names)
                    # 主从犯key-value对，key为被告人姓名，value为判决结果
                    principals, accomplices = {}, {}
                    # 除主从犯外的其他被告人
                    unpairable = []
                    for judgment in obj["outcomes"]:
                        if judgment["name"] in principal_names:
                            principals[judgment["name"]] = judgment
                        elif judgment["name"] in accomplice_names:
                            accomplices[judgment["name"]] = judgment
                        else:
                            unpairable.append(judgment)
                    # 大模型可能抽取出不属于被告人列表中的结果，需要筛除
                    principal_names = principals.keys()
                    accomplice_names = accomplices.keys()
                    if len(principal_names) == 0 or len(accomplice_names) == 0:
                        add_to_log("/home/hwh/my_multi_defendant/logs/data_logs/generate_pairs_data.log", f"case_id: {obj['case_id']} 抽取出不属于案件被告人列表的主/从犯")
                        continue
                    # 每个主犯配对一个从犯，剩余从犯两两配对
                    paired = [False] * len(accomplice_names)
                    for i, p in enumerate(principal_names):
                        already_paired = False
                        for j, a in enumerate(accomplice_names):
                            if not paired[j]:
                                if check_principal_accomplice_relation(principals[p]["judgment"], accomplices[a]["judgment"]):
                                    # 符合主从犯关系，可以pair
                                    pair_json = {"case_id": obj["case_id"], "original_fact": obj["fact"], "defendant1": p, "defendant2": a, "defendant1_fact": "", "defendant2_fact": "", "defendant1_outcome": principals[p]["judgment"], "defendant2_outcome": accomplices[a]["judgment"], "relation": "lead"}
                                    lead_pairs_num += 1
                                    paired[j] = True
                                    already_paired = True
                                    writer.write(pair_json)
                            if already_paired:
                                break
                        if not already_paired:
                            # 该主犯没有可以配对的从犯，与其它任意无法配对的被告人配对即可
                            unpairable.append(principals[p])
                    # 将还没配对的从犯加入unpairable
                    for j, a in enumerate(accomplice_names):
                        if not paired[j]:
                            unpairable.append(accomplices[a])        
                    randomly_pair_unrelated_defendants(obj, unpairable, writer)
    add_to_log("/home/hwh/my_multi_defendant/logs/data_logs/generate_pairs_data.log", f"成功配对为lead关系的pair数量：{lead_pairs_num}")

def fillin_extracted_fact_in_pairs(pair_file, extracted_fact_file):
    pair_file_saved = pair_file.removesuffix(".jsonl") + "_full.jsonl"
    print(f"file saved to {pair_file_saved}")
    case_defendant_fact = defaultdict(dict)
    run_log_file = "/home/hwh/my_multi_defendant/logs/data_logs/fill_in_extracted_fact_to_pairs.log"
    with jsonlines.open(extracted_fact_file, "r") as ef_reader:
        for obj in ef_reader:
            case_defendant_fact[obj["case_id"]][obj["defendant"]] = obj["extracted_fact"]
    with jsonlines.open(pair_file, "r") as p_reader:
        with jsonlines.open(pair_file_saved, "a") as writer:
            for obj in p_reader:
                if obj["case_id"] not in case_defendant_fact:
                    add_to_log(run_log_file, f"case {obj['case_id']} not extracted???")
                    continue
                if obj["defendant1"] not in case_defendant_fact[obj["case_id"]]:
                    add_to_log(run_log_file, f"case {obj['case_id']} defendant {obj['defendant1']} not extracted???")
                    continue
                if obj["defendant2"] not in case_defendant_fact[obj["case_id"]]:
                    add_to_log(run_log_file, f"case {obj['case_id']} defendant {obj['defendant2']} not extracted???")
                    continue
                obj["defendant1_fact"] = case_defendant_fact[obj["case_id"]][obj["defendant1"]]
                obj["defendant2_fact"] = case_defendant_fact[obj["case_id"]][obj["defendant2"]]
                writer.write(obj)


if __name__ == "__main__":
    # get_top_30_charges()
    # generate_joint_crime_pairs("/home/hwh/my_multi_defendant/data/curated/train_smaller_joint_crime_cases.jsonl", "/home/hwh/my_multi_defendant/data/curated/train_smaller_defendant_pairs.jsonl", lower_bound=85, upper_bound=None)
    # generate_non_joint_crimes_two_defendant_sample_pairs("/home/hwh/my_multi_defendant/data/cmdl/train_smaller_with_id.jsonl", "/home/hwh/my_multi_defendant/data/curated/train_smaller_defendant_pairs_non_jt.jsonl")
    # fillin_extracted_fact_in_pairs("/home/hwh/my_multi_defendant/data/curated/train_smaller_defendant_pairs_all.jsonl", "/home/hwh/my_multi_defendant/data/curated/train_smaller_extracted_fact.jsonl")
