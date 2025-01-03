import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    return parser.parse_args()