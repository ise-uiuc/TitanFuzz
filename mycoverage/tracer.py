# adapt from https://github.com/fuzzitdev/pythonfuzz/blob/master/pythonfuzz/tracer.py

import collections
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

prev_line = 0
prev_filename = ""
data = collections.defaultdict(set)
trace_library = None


def trace_tf(frame, event, arg):
    if event != "line":
        return trace_tf

    global prev_line
    global prev_filename

    func_filename = frame.f_code.co_filename
    func_line_no = frame.f_lineno

    if "tensorflow" not in func_filename:
        return trace_tf

    if func_filename != prev_filename:
        # We need a way to keep track of inter-files transferts,
        # and since we don't really care about the details of the coverage,
        # concatenating the two filenames in enough.

        # Why need to keep track of inter-file transfers?
        data[func_filename + prev_filename].add((prev_line, func_line_no))
    else:
        data[func_filename].add((prev_line, func_line_no))

    prev_line = func_line_no
    prev_filename = func_filename

    return trace_tf


def trace_torch(frame, event, arg):
    if event != "line":
        return trace_torch

    global prev_line
    global prev_filename

    func_filename = frame.f_code.co_filename
    func_line_no = frame.f_lineno

    if "torch" not in func_filename:
        return trace_torch

    if func_filename != prev_filename:
        # We need a way to keep track of inter-files transferts,
        # and since we don't really care about the details of the coverage,
        # concatenating the two filenames in enough.

        # Why need to keep track of inter-file transfers?
        data[func_filename + prev_filename].add((prev_line, func_line_no))
    else:
        data[func_filename].add((prev_line, func_line_no))

    prev_line = func_line_no
    prev_filename = func_filename

    return trace_torch


def get_coverage():
    return sum(map(len, data.values()))


def test_tf_tracer():
    sys.settrace(trace_tf)
    print("cov: ", get_coverage())
    try:
        import tensorflow as tf
    except Exception as e:
        print(e)
    print("cov: ", get_coverage())
    try:
        print(tf.add(2, 3))
    except Exception as e:
        print(e)
    print("cov: ", get_coverage())


def test_tf_tracer_after_import():
    print("cov: ", get_coverage())
    try:
        import tensorflow as tf
    except Exception as e:
        print(e)

    sys.settrace(trace_tf)
    print("cov: ", get_coverage())

    for _ in range(5):
        try:
            print(tf.add(2, 3))
        except Exception as e:
            print(e)
        print("cov: ", get_coverage())


if __name__ == "__main__":
    # test_tf_tracer()
    test_tf_tracer_after_import()
