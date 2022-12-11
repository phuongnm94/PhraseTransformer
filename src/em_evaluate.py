import argparse
import re

from nltk.translate.bleu_score import corpus_bleu


def norm_logic_form(in_str):
    in_str = re.sub(r'\$\w(\d)', r'$\1', in_str)
    return in_str


def acc(pred_lines, target_lines):
    assert len(pred_lines) == len(target_lines)
    count_right = 0
    for i, line in enumerate(pred_lines):
        if line == target_lines[i]:
            count_right += 1.0
        else:
            print(i)
            print(line)
            print(target_lines[i])
            print()
            print()
    print(count_right, len(pred_lines), count_right*1.0 / len(pred_lines))
    return count_right *1.0 / len(pred_lines)


def bleu(pred_lines, target_lines):
    assert len(pred_lines) == len(target_lines)
    references = [[x.split()] for x in target_lines]
    candidates = [x.split() for x in pred_lines]
    return corpus_bleu(references, candidates)


def run_all_metrics(path_base: str, pred_file_path: str, target_file_path: str):
    f_pred = open("{}/{}".format(path_base, pred_file_path), "rt")
    f_target = open("{}/{}".format(path_base, target_file_path), "rt")

    # process
    pred_lines = f_pred.readlines()
    target_lines = f_target.readlines()

    f_pred.close()
    f_target.close()

    # run metrics
    print('acc = ', acc(pred_lines, target_lines))
    print('bleu = ',bleu(pred_lines, target_lines))


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../data/atis',
                        required=False,
                        help="""Folder save  data.""")
    parser.add_argument('--pred', default='Y_pred_9.tsv',
                        required=False,
                        help="""result predict""")
    parser.add_argument('--target', default='Y_test_9.tsv',
                        required=False,
                        help="""target value""")

    opt, unknown = parser.parse_known_args()
    run_all_metrics(opt.path, pred_file_path=opt.pred, target_file_path=opt.target)
