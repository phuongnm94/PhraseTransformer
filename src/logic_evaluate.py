import argparse
import logging

from logical_utils.tree import STree, is_tree_eq


def log(msg, print_stdout=True):
    if print_stdout:
        print(msg)
    logging.info(msg)


def f1(p, r):
    if p+r == 0:
        return 0
    else:
        return 2*p*r/(p+r)

def corpus_sent_acc(preds, targets):

    method_eq = is_tree_eq
    obj_parser = STree 
    count_all = len(preds)

    count_logic = 0
    count_exact_matching = 0
    for i, logic1 in enumerate(targets):
        logic2 = preds[i]
        try:
            if logic2 == logic1:
                count_exact_matching += 1.0
            elif method_eq(obj_parser(logic1), obj_parser(logic2), not_layout=True):
                count_logic += 1.0
            else:
                pass
        except:
            pass 
    return (count_exact_matching + count_logic) *100.0 / count_all

def word_level_acc(pred_file, test_file):
    with open(pred_file, "rt", encoding="utf8") as f:
        pred = [l.strip().split(" ") for l in f.readlines()]
    with open(test_file, "rt", encoding="utf8") as f:
        test = [l.strip().split(" ") for l in f.readlines()]

    count_s_right = 0
    count_right = 0
    count_pred = 0
    count_gold = 0
    for i, l in enumerate(pred):
        if set(l) == set(test[i]):
            count_s_right += 1
        count_right += len(set(l).intersection(set(test[i])))
        count_pred += len(set(l))
        count_gold += len(set(test[i]))
    log("count (w): correct={}, pred={}, gold={}".format(
        count_right, count_pred, count_gold))
    log("p={}, r={}, f1={}".format(count_right/count_pred, count_right/count_gold,
                                   f1(count_right/count_pred, count_right/count_gold)))
    log("count(s): correct={}, gold={}, acc={}".format(
        count_s_right, len(pred), count_s_right/len(pred)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluate accuracy of logical form.')
    parser.add_argument('--path',
                        default="../data/geo/8536_seq2seq_marking_10000step", )
    parser.add_argument('--src',
                        default="X_test_5.tsv", )
    parser.add_argument('--tgt',
                        default="Y_test_5.tsv", )
    parser.add_argument('--pred',
                        default="Y_pred_5.tsv", )
    parser.add_argument('--type', required=False,
                        default="logic",
                        help="Choose type to run method compare in [logic|code|wordacc]")
    parser.add_argument('--n_best', required=False, type=int,
                        default=1,
                        help="n best prediction by beam-search")
    args = parser.parse_args()
    print(args.path)

    logging.basicConfig(level=logging.DEBUG, filename="{}/result_logic.log".format(args.path), filemode="w+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if args.type == "wordacc":
        word_level_acc("{}/{}".format(args.path, args.pred),
                       "{}/{}".format(args.path, args.tgt))
    else:
        method_eq = is_tree_eq
        obj_parser = STree
        if "django" in args.path or args.type == "code":
            method_eq = is_code_eq
            obj_parser = SCode

        data = {}
        for file_name in [args.src, args.tgt, args.pred]:
            with open("{}/{}".format(args.path, file_name), "rt", encoding="utf8") as f:
                lines = [l.strip() for l in f.readlines()]
                data[file_name] = lines
        count_all = len(data[args.src])

        count_logic = 0
        count_exact_matching = 0
        count_nbest = 0
        for i, logic1 in enumerate(data[args.tgt]):
            for j in range(args.n_best):
                logic2 = data[args.pred][i * args.n_best + j]
                if j == 0:
                    try:
                        if logic2 == logic1:
                            count_exact_matching += 1.0
                            break
                        elif method_eq(obj_parser(logic1), obj_parser(logic2), not_layout=True):
                            count_logic += 1.0
                            log("++ sentence {}, logic/code gold == logic/code pred ++".format(i))
                            log(logic1)
                            log(logic2)
                            break
                        else:
                            log("-- sentence {}, logic/code gold != logic/code pred --".format(i))
                            log(data[args.src][i])
                            log(logic1)
                            log(logic2)
                    except:
                        pass
                else:
                    if (logic1 == logic2 or (
                            method_eq(obj_parser(logic1), obj_parser(logic2), not_layout=True))):
                        log("!! sentence {}, logic/code gold == logic/code pred {} !!".format(i, j + 1))
                        log(data[args.src][i])
                        log(logic1)
                        log(logic2)
                        count_nbest += 1.0
                        break

        log(
            "exact match acc: {}, {}, {}".format(count_exact_matching, count_all, count_exact_matching / count_all))
        log(
            "logic/code acc : {}, {}, {}".format(count_logic, count_all, (count_exact_matching + count_logic) / count_all))
        log("logic/code using nbest = {}, {}, {}, {}".format(args.n_best, count_nbest, count_all,
                                                             (count_exact_matching + count_nbest
                                                              + count_logic) / count_all))
