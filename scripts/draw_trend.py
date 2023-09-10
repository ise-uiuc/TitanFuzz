import os

from brokenaxes import brokenaxes
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

os.makedirs("figs", exist_ok=True)
plt.rcParams["font.family"] = "monospace"
plt.rcParams["axes.linewidth"] = 1.2


tf = """0       36      45      31      30      33      27
0.1     326     441     221     46      54      41
0.2     518     622     381     52      58      46
0.3     582     667     365     49      61      54
0.4     527     669     330     53      64      54
0.5     525     585     280     55      63      57
0.6     441     559     233     56      64      53
0.7     395     500     176     56      62      51
0.8     354     415     154     56      63      48
0.9     254     373     123     52      62      44
1       201     305     123     45      55      40
"""

pt = """0       70      82      15      56      69      10
0.1     733     862     116     75      80      36
0.2     1119    1382    152     76      82      53
0.3     1194    1517    208     78      86      64
0.4     1157    1464    201     82      85      65
0.5     1091    1416    187     84      87      61
0.6     966     1291    143     86      86      61
0.7     900     1177    152     83      86      58
0.8     783     1045    130     81      85      58
0.9     661     898     107     84      85      55
1       617     817     96      84      84      53
"""

pt_coverage = """0      6       12      18      24      30      36      42      48      54      60
15607   16484   16486   16490   16490   16490   16490   16490   16490   16492   16492
23231   23327   23377   23471   23471   23522   23528   23775   23808   23819   23823"""

tf_coverage = """0      6       12      18      24      30      36      42      48      54      60
79145   84426   84493   84534   84553   84594   84606   84608   84610   84616   84618
104926  105054  106458  106794  106950  107091  107264  107370  107490  107603  107685"""


def get_coverage_data():
    lines = tf_coverage.splitlines()
    tf_result = {
        "time": [float(x) for x in lines[0].split()],
        "DeepRel": [float(x) for x in lines[1].split()],
        "TitanFuzz": [float(x) for x in lines[2].split()],
    }
    lines = pt_coverage.splitlines()
    pt_result = {
        "time": [float(x) for x in lines[0].split()],
        "DeepRel": [float(x) for x in lines[1].split()],
        "TitanFuzz": [float(x) for x in lines[2].split()],
    }
    return tf_result, pt_result


def get_graph_data():
    tf_result = {
        "temp": [],
        "API (unique)": [],
        "API-Signature (unique)": [],
        "no-step (unique)": [],
        "API (coverage)": [],
        "API-Signature (coverage)": [],
        "no-step (coverage)": [],
    }
    for line in tf.splitlines():
        s = line.split()
        tf_result["temp"].append(float(s[0]))
        tf_result["API (unique)"].append(float(s[1]))
        tf_result["API-Signature (unique)"].append(float(s[2]))
        tf_result["no-step (unique)"].append(float(s[3]))
        tf_result["API (coverage)"].append(float(s[4]))
        tf_result["API-Signature (coverage)"].append(float(s[5]))
        tf_result["no-step (coverage)"].append(float(s[6]))

    pt_result = {
        "temp": [],
        "API (unique)": [],
        "API-Signature (unique)": [],
        "no-step (unique)": [],
        "API (coverage)": [],
        "API-Signature (coverage)": [],
        "no-step (coverage)": [],
    }

    for line in pt.splitlines():
        s = line.split()
        pt_result["temp"].append(float(s[0]))
        pt_result["API (unique)"].append(float(s[1]))
        pt_result["API-Signature (unique)"].append(float(s[2]))
        pt_result["no-step (unique)"].append(float(s[3]))
        pt_result["API (coverage)"].append(float(s[4]))
        pt_result["API-Signature (coverage)"].append(float(s[5]))
        pt_result["no-step (coverage)"].append(float(s[6]))

    return tf_result, pt_result


def graph_codex_temperature():
    tf_result, pt_result = get_graph_data()
    fig, ax = plt.subplots(2, 2, figsize=(7.5, 4))
    ax[0, 0].plot(tf_result["temp"], tf_result["API-Signature (unique)"])
    ax[0, 0].plot(tf_result["temp"], tf_result["API (unique)"])
    ax[0, 0].plot(tf_result["temp"], tf_result["no-step (unique)"])
    ax[0, 0].axvline(x=0.4, color="r", linestyle="--")
    ax[0, 0].set(ylabel="# valid programs")
    ax[1, 0].plot(tf_result["temp"], tf_result["API-Signature (coverage)"])
    ax[1, 0].plot(tf_result["temp"], tf_result["API (coverage)"])
    ax[1, 0].plot(tf_result["temp"], tf_result["no-step (coverage)"])
    ax[1, 0].axvline(x=0.4, color="r", linestyle="--")
    ax[1, 0].set(ylabel="# of covered APIs")
    ax[1, 0].set(xlabel="temperature")
    ax[0, 0].set_title("TensorFlow")

    ax[0, 1].plot(pt_result["temp"], pt_result["API-Signature (unique)"])
    ax[0, 1].plot(pt_result["temp"], pt_result["API (unique)"])
    ax[0, 1].plot(pt_result["temp"], pt_result["no-step (unique)"])
    ax[0, 1].axvline(x=0.4, color="r", linestyle="--")

    ax[1, 1].plot(
        pt_result["temp"], pt_result["API-Signature (coverage)"], label="TitanFuzz"
    )
    ax[1, 1].plot(
        pt_result["temp"], pt_result["API (coverage)"], label="TitanFuzz-sig."
    )
    ax[1, 1].plot(
        pt_result["temp"], pt_result["no-step (coverage)"], label="TitanFuzz-step"
    )
    ax[1, 1].axvline(x=0.4, color="r", linestyle="--")
    ax[1, 1].legend(loc="lower right")
    ax[1, 1].set(xlabel="temperature")
    ax[0, 1].set_title("PyTorch")
    plt.tight_layout()
    plt.savefig("figs/codex_temperature.png")


def graph_coverage_trend():
    tf_result, pt_result = get_coverage_data()
    fig = plt.figure(figsize=(12, 3.5))
    sps1, sps2 = GridSpec(1, 2)  # , hspace=0.3)

    bax = brokenaxes(
        ylims=((78800, 85000), (104500, 108000)), subplot_spec=sps1, despine=False
    )
    bax.plot(
        tf_result["time"],
        tf_result["DeepRel"],
        label="DeepRel",
        linewidth=3,
        marker="v",
        markersize=8,
        linestyle="dashed",
    )
    bax.plot(
        tf_result["time"],
        tf_result["TitanFuzz"],
        label="TitanFuzz",
        linewidth=3,
        marker="o",
        markersize=8,
        linestyle="dashed",
    )

    bax.set_title("TensorFlow")
    bax.legend(loc=4)
    bax.set_ylabel("Line Coverage", labelpad=35)
    bax.set_xlabel("time (seconds)")
    bax = brokenaxes(
        ylims=((15500, 16700), (23000, 24000)), subplot_spec=sps2, despine=False
    )
    bax.plot(
        pt_result["time"],
        pt_result["DeepRel"],
        label="DeepRel",
        linewidth=3,
        marker="v",
        markersize=8,
        linestyle="dashed",
    )
    bax.plot(
        pt_result["time"],
        pt_result["TitanFuzz"],
        label="TitanFuzz",
        linewidth=3,
        marker="o",
        markersize=8,
        linestyle="dashed",
    )

    bax.set_title("PyTorch")

    bax.legend(loc=4)
    bax.set_xlabel("time (seconds)")
    bax.set_ylabel("Line Coverage", labelpad=35)
    plt.savefig("figs/coverage_trend.png")


graph_codex_temperature()
graph_coverage_trend()
