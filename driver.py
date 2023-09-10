import argparse
import os
import subprocess
import time

from util import util


def clean_cache():
    print("Cleaning cache...")
    try:
        subprocess.run(
            "lsof /dev/nvidia* | grep 'chenyuan' | grep 'python' | grep 'nvidia0' | awk '{print $2}' | xargs -I {} kill {}",
            shell=True,
            timeout=10,
        )
    except Exception as e:
        print("Error when cleaning cache")
        print(e)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--car", type=str, default="torch2cuda.py"
    )  # car is something that is driven by a driver
    parser.add_argument(
        "--tf", action="store_true", default=False
    )  # tensorflow or torch
    parser.add_argument("--mode", type=str, default="race")  # mode to be passed to car
    parser.add_argument("--input", type=str, default=None)  # source directory
    parser.add_argument("--output", type=str, default="trace.txt")  # output filename
    parser.add_argument(
        "--cont", action="store_true", default=False
    )  # continue on a trace in the middle
    args = parser.parse_args()

    tasks = util.readAllTasksFromDir(args.input)
    nTasks = len(tasks)

    if not args.cont:
        catchesLog = open("catches.log", "w")
        catchesLog.write(args.input + " " + str(nTasks) + "\n")
        catchesLog.close()
        trace = open(args.output, "w")
        trace.write(args.input + " " + str(nTasks) + "\n")
        trace.close()
    trace = open(args.output, "r")

    cur: int = 0
    # Support for manually re-booting in the middle
    for line in trace.readlines():
        id, api, label, state = util.parseResultSummary(line)
        if id == None:
            continue
        cur = id + 1
    trace.close()
    trace = open(args.output, "a")

    lastFailApi: str = ""
    continuousTimeouts: int = 0
    cnt_OOM = 0

    while cur < nTasks:
        proc = None
        try:
            print("\nDriver: Car Reboot", file=trace)
            trace.flush()

            api, label, src = util.parseTask(tasks[cur])
            if continuousTimeouts >= 5:
                while api == lastFailApi:
                    cur += 1
                    print(
                        "\nTitanFuzzTestcase",
                        id,
                        api,
                        label,
                        "TimeoutSkipped",
                        "no detail",
                        file=trace,
                    )
                    trace.flush()
                    api, label, src = util.parseTask(tasks[cur])
                continuousTimeouts = 0
                continue

            cmdLine = [
                "python",
                "-u",
                args.car,
                "--mode",
                args.mode,
                "--input",
                args.input,
                "--start",
                str(cur),
                "--singleapi",
            ]
            if args.tf:
                cmdLine.append("--tf")
            proc = subprocess.Popen(
                cmdLine, stdin=subprocess.PIPE, stdout=trace, stderr=subprocess.STDOUT
            )

            # Monitor the size of output file to see whether subprocess is stuck
            noNewOutputSecs = 0
            oldFileSize = 0
            while proc.poll() == None and noNewOutputSecs < 20:
                time.sleep(1)
                fileSize = os.path.getsize(args.output)
                if oldFileSize != fileSize:
                    if noNewOutputSecs >= 10:
                        print("Got some output, timer reset")
                    noNewOutputSecs = 0
                else:
                    noNewOutputSecs += 1
                    if noNewOutputSecs >= 10:
                        print(noNewOutputSecs, "seconds without new output")
                oldFileSize = fileSize
            metTimeout = False
            if noNewOutputSecs >= 20:
                print("Timeout, killed")
                metTimeout = True

            while proc.poll() is None:
                proc.kill()

            # Try to find how many testcases have been covered
            trace.close()
            trace = open(args.output, "r")
            lastId: int = 0
            cur_OOM = 0
            for line in trace.readlines():
                if "CUDA error: out of memory" in line:
                    cur_OOM += 1
                id, api, label, state = util.parseResultSummary(line)
                if id == None:
                    continue
                lastId = id

            trace.close()
            if cur_OOM > cnt_OOM + 20:
                cnt_OOM = cur_OOM
                clean_cache()
            trace = open(args.output, "a")

            reason = ""
            if metTimeout:  # Increase cursor and print timeout catch
                cur = max(lastId + 2, cur + 1)  # Skip the one causing timeout
                reason = "TimeoutFail"
                continuousTimeouts += 1
                print("continuousTimeouts", continuousTimeouts)
            elif proc.returncode != 233:  # FrameworkCrashCatch
                cur = max(lastId + 2, cur + 1)
                reason = "FrameworkCrashCatch"
                continuousTimeouts += 1
            else:  # no problem
                cur = lastId + 1
                continuousTimeouts = 0
                continue
            if cur >= nTasks:
                break
            failId = cur - 1
            failTask = tasks[failId]
            failApi = failTask[0]
            failLabel = failTask[1]
            print(
                "\nTitanFuzzTestcase",
                failId,
                failApi,
                failLabel,
                reason,
                "no detail",
                file=trace,
            )
            trace.flush()
            lastFailApi = failApi

        except KeyboardInterrupt:
            proc.kill()  # Safely kill subprocess
            exit(-1)

    # Statistics

    trace.close()
    trace = open(args.output, "r")
    resultStat = {}
    caughtApiCount = 0
    testedCount = 0
    lastCaughtApi = ""
    catches = []
    for line in trace.readlines():
        id, api, label, state = util.parseResultSummary(line)
        if id == None:
            continue
        testedCount += 1
        if state not in resultStat:
            resultStat[state] = 0
        resultStat[state] += 1
        if "Catch" in state and api != lastCaughtApi:
            caughtApiCount += 1
            lastCaughtApi = api
            catches.append([api, label, state])

    trace.close()
    trace = open(args.output, "a")

    print("==== driven race finished ====", file=trace)
    print("total", len(tasks), file=trace)
    print("tested", testedCount, file=trace)
    for reason, count in resultStat.items():
        print(reason, count, file=trace)

    trace.close()

    return 0


if __name__ == "__main__":
    exit(main())
