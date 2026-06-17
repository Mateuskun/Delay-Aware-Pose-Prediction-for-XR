#!/usr/bin/env python

import json
import argparse
from collections import OrderedDict


def json_difference(obj1, obj2, r=False):
    assert type(obj1) == type(obj2)

    if isinstance(obj1, dict):
        assert obj1.keys() == obj2.keys()
        return {k: json_difference(obj1[k], obj2[k], r) for k in obj1.keys()}

    if isinstance(obj1, list):
        assert len(obj1) == len(obj2)
        return [json_difference(e1, e2, r) for e1, e2 in zip(obj1, obj2)]

    if isinstance(obj1, (int, float)):
        return obj1 - obj2 if not r else f"{(obj1 - obj2) / obj1 * 100:.2f}%"

    return obj1


def main():

    parser = argparse.ArgumentParser(
        description="Compute numerical differences between two JSON files with the same structure"
    )
    parser.add_argument("input_file1", type=str, help="Path to the first input JSON file")
    parser.add_argument("input_file2", type=str, help="Path to the second input JSON file")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file")
    parser.add_argument(
        "--use_relative",
        "-r",
        dest="use_relative",
        action="store_true",
        help="Whether to show relative percentage diff instead of absolute diff",
    )

    args = parser.parse_args()

    input_file1 = args.input_file1
    input_file2 = args.input_file2
    output_file = args.output_file
    use_relative = args.use_relative

    with open(input_file1, "r") as f1, open(input_file2, "r") as f2:
        data1 = json.load(f1, object_pairs_hook=OrderedDict)
        data2 = json.load(f2, object_pairs_hook=OrderedDict)

    result = json_difference(data1, data2, use_relative)

    print(json.dumps(result, indent=4))
    with open(output_file, "w") as output_f:
        json.dump(result, output_f, indent=4)

    print("Differences calculated and saved to", output_file)


if __name__ == "__main__":
    main()
