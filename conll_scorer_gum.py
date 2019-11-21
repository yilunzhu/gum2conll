import io, os, sys
import re
import tempfile
import subprocess
from gum2conll import read_conll_file

COREF_RESULTS_REGEX = re.compile(
    r".*Coreference: Recall: \(([0-9.]+) / ([0-9.]+)\) [0-9.]+%\tPrecision: \(([0-9.]+) / ([0-9.]+)\) [0-9.]+%\tF1: [0-9.]+%.*",
    re.DOTALL)


def read_dir(dir_path, gold_files):
    filenames = []
    for filename in os.listdir(dir_path):
        file_path = dir_path + os.sep + filename
        gold_file_path = gold_files + os.sep + filename
        if filename.endswith("conll"):
            filenames.append((file_path, gold_file_path))
    return filenames


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=False):
    cmd = ["/home/yz565/coref/bert_coref/conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall_correct = float(coref_results_match.group(1))
    recall_total = float(coref_results_match.group(2))
    precision_correct = float(coref_results_match.group(3))
    precision_total = float(coref_results_match.group(4))
    return recall_correct, recall_total, precision_correct, precision_total


def evaluate_conll(gold_path, predictions, subtoken_maps, official_stdout=False):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as prediction_file:
        with open(gold_path, "r", encoding="utf-8") as gold_file:
            output_conll(gold_file, prediction_file, predictions, subtoken_maps)
        print("Predicted conll file: {}".format(prediction_file.name))
    return {m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in
            ("muc", "bcub", "ceafe")}


def main(xrenner_files, gold_files):
    filenames = read_dir(xrenner_files, gold_files)

    sum_f = 0
    dic = {}

    for metric in ["muc", "bcub", "ceafe"]:
        sum_recall_correct = 0
        sum_recall_total = 0
        sum_precision_correct = 0
        sum_precision_total = 0
        for (pred_file, gold_file) in filenames:
            print(pred_file, gold_file)
            pred_filename = pred_file.split("/")[-1]
            gold_filename = gold_file.split("/")[-1]
            pred = read_conll_file(pred_file)
            gold = read_conll_file(gold_file)

            with io.open(xrenner_files + os.sep + "singleton" + os.sep + pred_filename, "w", encoding="utf-8") as f:
                name = pred_file.split("/")[-1].split(".")[0]
                f.write(f"#begin document {name}\n")
                for line in pred:
                    line = line.replace("|", "")
                    f.write(f"{line}\n")
                f.write("#end document\n")
            with io.open(gold_files + os.sep + "singleton" + os.sep + gold_filename, "w", encoding="utf-8") as f:
                name = gold_file.split("/")[-1].split(".")[0]
                f.write(f"#begin document {name}\n")
                for line in gold:
                    line = line.replace("|", "")
                    f.write(f"{line}\n")
                f.write("#end document\n")
            pred_file_new = xrenner_files + os.sep + "singleton" + os.sep + pred_filename
            gold_file_new = gold_files + os.sep + "singleton" + os.sep + gold_filename

            recall_correct, recall_total, precision_correct, precision_total = official_conll_eval(gold_file_new,
                                                                                                   pred_file_new,
                                                                                                   metric)
            sum_recall_correct += recall_correct
            sum_recall_total += recall_total
            sum_precision_correct += precision_correct
            sum_precision_total += precision_total
        recall = sum_recall_correct / sum_recall_total
        precision = sum_precision_correct / sum_precision_total
        f1 = 2 * recall * precision / (recall + precision)
        a = {"p": precision, "r": recall, "f": f1}
        print(str(a))
        dic[metric] = a
        sum_f += f1

    avg_f1 = sum_f / 3
    print(str(dic))
    print(str(avg_f1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gum":
        xrenner_files = "xrenner"
        gold_files = "gum" + os.sep + "coref" + os.sep + "conll"
        main(xrenner_files, gold_files)
    elif mode == "gumby":
        xrenner_files = "xrenner_gumby"
        gold_files = "gumby_gold" + os.sep + "conll"
        main(xrenner_files, gold_files)
    else:
        "You can add your mode here."
        sys.stderr.write("unrecognized mode. Please enter either gum or gumby.")
        sys.exit()
