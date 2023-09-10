import argparse
import os

from util import util


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="trace.txt"
    )  # trace from previous runs of driver.py
    parser.add_argument(
        "--output", type=str, default="trace-compressed.txt"
    )  # trace from previous runs of driver.py
    args = parser.parse_args()

    trace = open(args.input, "r")
    compressed = open(args.output, "w")
    for line in trace.readlines():
        id, api, label, state = util.parseResultSummary(line)
        if id != None:
            compressed.write(line)
    trace.close()
    compressed.close()


if __name__ == "__main__":
    main()
