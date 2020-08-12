import sys, json
sys.path.append("../..")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats


def run(results_path = "node1/1597073762/step1_metrics.json"):
    #results_path = "../output/step1_metrics.json"
    mutants_original = json.load(open("../output/metrics/{}".format(results_path)))

    mutants = [m for m in mutants_original if len(m["mutations"][0])>0]
    mutants_names = [m["mutant"] for m in mutants]
    mutants_attrs = [str(m["mutations"][0][0]["mutation_target"].split("#")) for m in mutants]
    print(mutants_attrs)
    mutants_names = mutants_attrs

    train_accuracies_step = np.array([m["train_acc"] for m in mutants])
    test_accuracies_step = np.array([m["test_acc"] for m in mutants])

    df_steps_train = pd.DataFrame(data=train_accuracies_step.swapaxes(0,1), columns=mutants_names)
    df_steps_test = pd.DataFrame(data=test_accuracies_step.swapaxes(0, 1), columns=mutants_names)
    df_steps_train.plot(figsize=(20,15), title="train accuracy {}".format(results_path))
    df_steps_test.plot(figsize=(20, 15), title="test accuracy {}".format(results_path))


def run_adv(results_path = "node1/1597073762/step1_metrics.json"):

    mutants_original = json.load(open("../output/metrics/{}".format(results_path)))

    mutants = [m for m in mutants_original if len(m["mutations"][0])>0]

    test_accuracies = [m["robustness_score"][1] if isinstance(m["robustness_score"], list) else 0 for m in mutants]
    adversarial_accuracies = [m["robustness_score"][2] if isinstance(m["robustness_score"], list) else 0 for m in
                              mutants]

    df_accuracy = pd.DataFrame(np.array([test_accuracies,adversarial_accuracies]).swapaxes(0,1), columns=["test","adv"])
    labels = [str(m["mutations"][0][0]["mutation_target"].split("#")) for m in mutants]
    df_accuracy.index = labels
    print(df_accuracy.index)
    ax = df_accuracy.plot(figsize=(20, 15), x_compat=True, title="test/adv accuracy {}".format(results_path))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels,rotation=40)


run("node1/1597073762/step1_metrics.json")
run("node8/1597081929/step1_metrics.json")
run_adv("local_node8_noAugment/1597155193/step1_metrics.json")
run_adv("local_node1_noAugment/1597221442/step1_metrics.json")


#run("node8_noAugment/1597159542/step1_metrics.json")
#run_adv("node8_noAugment/1597159542/step1_metrics.json")
plt.show()




