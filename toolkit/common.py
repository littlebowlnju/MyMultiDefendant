import jsonlines

def add_to_log(file, log_str):
    with open(file, "a") as f:
        f.write(log_str + "\n")
    print(log_str)
    return

def merge_two_jl_files(file1, file2, merged_file):
    with jsonlines.open(merged_file, "a") as writer:
        with jsonlines.open(file1, "r") as reader:
            for l in reader:
                writer.write(l)
        with jsonlines.open(file2, "r") as reader:
            for l in reader:
                writer.write(l)