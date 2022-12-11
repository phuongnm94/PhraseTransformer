from onmt.translate import Translator

print(' '.join(Translator.merge_sentence_label([
    "O", "B-x", "I-x", "O", "B-g", "I-g"
], [
    "a", "b", "c", "ax", "bx", "cx"
], ["x", "_b","x", "_a", "g"])))
