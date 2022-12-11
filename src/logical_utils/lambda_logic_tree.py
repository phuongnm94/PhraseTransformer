import copy
import logging
import re
from typing import List

logger = logging.getLogger()


class DifferenceCode:
    PREDICATE = 0
    VARIABLE = 1
    CONSTANT = 2
    CHILD_COUNT = 3


class LogicElement:
    DEFAULT_RELAX_CHILD_ORDER = {"and", "or", "", "next_to"}
    DEFAULT_ALLOW_CHILD_DUPLICATION = {"and", "or", ""}

    def __init__(self, value="", child=None, relax_child_order=False, allow_child_duplication=False,
                 is_hierarchical=False):
        self.is_hierarchical = is_hierarchical
        self.child = child or []
        self.value = str(value)
        if value in LogicElement.DEFAULT_RELAX_CHILD_ORDER:
            relax_child_order = True
        self.relax_child_order = relax_child_order

        if value in LogicElement.DEFAULT_ALLOW_CHILD_DUPLICATION:
            allow_child_duplication = True
        self.allow_child_duplication = allow_child_duplication

        self.leaf_nodes = None

    def add_child(self, child):
        if isinstance(child, LogicElement):
            self.child.append(child)
        else:
            logger.warning("Can't add child that is not object LogicElement: {}" % {child})

    def is_variable_node(self, term_check=r'[\$\?][^\s\n]*'):
        if len(self.child) == 0 and re.fullmatch(term_check, self.value):
            return True
        else:
            return False

    def is_constant(self, term_check_variable=r'[\$\?][^\s\n]*'):
        if len(self.child) == 0 and not self.is_variable_node(term_check_variable):
            return True
        else:
            return False

    def is_triple(self):
        if len(self.child) > 0 and len(self.value) > 0:
            return True
        else:
            return False

    def is_leaf_node(self):
        # if node that have nephew node
        for ee in self.child:
            if isinstance(ee, LogicElement) and len(ee.child) > 0:
                return False
        return True

    def get_leaf_nodes(self):
        tmp_leaf_nodes = []
        if self.leaf_nodes is not None:
            return self.leaf_nodes
        elif self.is_leaf_node():
            tmp_leaf_nodes = [self]
        elif self.leaf_nodes is None:
            for e in self.child:
                tmp_leaf_nodes += e.get_leaf_nodes()
        self.leaf_nodes = tmp_leaf_nodes
        return tmp_leaf_nodes

    def get_triple_name(self):
        tmp_triple_name = []
        if self.is_triple():
            tmp_triple_name.append(self.value)

        if len(self.child) > 0:
            for e in self.child:
                tmp_triple_name += e.get_triple_name()

        return tmp_triple_name

    def get_constant(self):
        tmp_triple_name = []
        if self.is_constant():
            return [self.value]
        for e in self.child:
            tmp_triple_name += e.get_constant()

        return tmp_triple_name

    @staticmethod
    def _collapse_list_logic(logics: List) -> List:
        len_logic = len(logics)
        mask_remove = [False] * len_logic
        for i in range(len_logic - 1):
            for j in range(i + 1, len_logic):
                if not mask_remove[j]:
                    if logics[i] == logics[j]:
                        mask_remove[j] = True
        for j in range(len_logic - 1, -1, -1):
            if mask_remove[j]:
                logics.pop(j)

        return logics

    def __eq__(self, other):
        if not isinstance(other, LogicElement) or not self.value == other.value:
            return False
        if self.is_hierarchical:
            check = True
            for c in self.child:
                if len(c.child) == 0:
                    check = False
                    break
            self.relax_child_order = check

        if self.allow_child_duplication and self.relax_child_order:
            self_child = copy.deepcopy(self.child)
            other_child = copy.deepcopy(other.child)

            self_child = self._collapse_list_logic(self_child)
            other_child = self._collapse_list_logic(other_child)
        else:
            self_child = self.child
            other_child = other.child

        if not len(self_child) == len(other_child):
            return False
        else:
            if not self.relax_child_order:
                for i, e in enumerate(other_child):
                    if not e == self_child[i]:
                        return False
            else:
                for i, e in enumerate(other_child):
                    j = 0
                    for _, e2 in enumerate(self_child):
                        if e == e2:
                            break
                        j += 1
                    if j == len(self_child):
                        return False
            return True

    def get_path_to_leaf_nodes(self, path_to_leaf_nodes=None, cur_path=None):
        path_to_leaf_nodes = [] if path_to_leaf_nodes is None else path_to_leaf_nodes
        cur_path = [] if cur_path is None else copy.deepcopy(cur_path)
        if len(self.value) > 0:
            cur_path.append(self.value)
        if self.is_leaf_node():
            path_to_leaf_nodes.append(cur_path)
        else:
            for e in self.child:
                e.get_path_to_leaf_nodes(path_to_leaf_nodes, cur_path)
        return path_to_leaf_nodes

    def __str__(self):
        separate_logic_1 = "[" if self.is_hierarchical else "("
        separate_logic_2 = "]" if self.is_hierarchical else ")"
        if len(self.child) == 0:
            return self.value
        child_str = " ".join([str(v) for v in self.child])
        if len(self.value) > 0 and len(child_str) > 0:
            return "{} {} {} {}".format(separate_logic_1, self.value, child_str, separate_logic_2)
        elif len(self.value) > 0:
            return "{} {} {}".format(separate_logic_1, self.value, separate_logic_2)
        elif len(child_str) > 0:
            return "{} {} {}".format(separate_logic_1, child_str, separate_logic_2)
        else:
            return "{} {}".format(separate_logic_1, separate_logic_2)

    @staticmethod
    def _norm_variable_name(name):
        if "$" in name:
            name = name.replace("$", "s")
        if "?" in name:
            name = name.replace("?", "") + "1"
        return name

    @staticmethod
    def _norm_predicate(name):
        if name == "":
            name = 'rootNode'
        name = name.replace(">", "Greater")
        name = name.replace("<", "Less")
        name = name.replace("<", "Less")
        return re.sub(r'[\.:_]', "-", name)

    @staticmethod
    def _norm_constant(value):
        value = re.sub(r'[\"]', "-", value)
        # value = re.sub(r'[^a-zA-Z\d]', "-", value)
        if re.search(u'[\u4e00-\u9fff]', value):
            value = "chinese-" + re.sub(u'[^a-zA-Z\d]', '-', value)
        value = "\"{}\"".format(value)
        return value

    def to_amr(self, var_exist=None):
        var_exist = var_exist or {}

        if self.is_constant():
            return self._norm_constant(self.value)
        elif self.is_variable_node():
            if var_exist is not None and self.value not in var_exist:
                var_exist[self.value] = self._norm_variable_name(self.value)
                return "({} / var)".format(self._norm_variable_name(self.value))
            else:
                return self._norm_variable_name(self.value)
        else:
            if self.value == "" and len(self.child) == 1:
                amr_str = self.child[0].to_amr(var_exist)
                if amr_str == "\"\"" or amr_str == "":
                    return "(n0 / errorRootNode)"
                elif re.fullmatch(r'".+"', amr_str):
                    return "(n0 / {})".format(amr_str)
                else:
                    return amr_str

            node_name = 'n' + str(len(var_exist))
            amr_str = "({} / {} ".format(node_name, self._norm_predicate(self.value))
            var_exist[node_name] = self._norm_predicate(self.value)
            for i, child in enumerate(self.child):
                child_amr = " :ARG{} ".format(i) + child.to_amr(var_exist)
                amr_str += child_amr
            amr_str += ")"
            return amr_str


def parse_lambda(logic_str: str):
    lg_parent = LogicElement()
    tk_arr = logic_str.split()
    tk_arr = [tk for tk in tk_arr if tk != "," and len(tk) > 0]
    tmp_logic = [lg_parent]
    j = 0
    for i in range(len(tk_arr)):
        if i + j >= len(tk_arr):
            break
        tk = tk_arr[i + j]
        if tk == "(":
            if i + j + 1 < len(tk_arr) and not tk_arr[i + j + 1] == "(":
                new_lg = LogicElement(value=tk_arr[i + j + 1])
                j += 1
            else:
                new_lg = LogicElement()
            tmp_logic[-1].add_child(new_lg)
            tmp_logic.append(new_lg)
        elif tk == ")":
            tmp_logic.pop()
        else:
            tmp_logic[-1].add_child(LogicElement(value=tk))

    return lg_parent if len(lg_parent.child) > 1 else lg_parent.child[0]


def parse_hierarchical_logic(logic_str: str):
    lg_parent = LogicElement(is_hierarchical=True)
    tk_arr = logic_str.replace("[", " [ ").split()
    tk_arr = [tk for tk in tk_arr if tk != "," and len(tk) > 0]
    tmp_logic = [lg_parent]
    j = 0
    for i in range(len(tk_arr)):
        if i + j >= len(tk_arr):
            break
        tk = tk_arr[i + j]
        if tk == "[":
            if i + j + 1 < len(tk_arr) and not tk_arr[i + j + 1] == "(":
                new_lg = LogicElement(value=tk_arr[i + j + 1], is_hierarchical=True)
                j += 1
            else:
                new_lg = LogicElement(is_hierarchical=True)
            tmp_logic[-1].add_child(new_lg)
            tmp_logic.append(new_lg)
        elif tk == "]":
            tmp_logic.pop()
        else:
            tmp_logic[-1].add_child(LogicElement(value=tk, is_hierarchical=True))

    return lg_parent if len(lg_parent.child) > 1 else lg_parent.child[0]


def parse_prolog(logic_str: str):
    lg_parent = LogicElement()
    tk_arr = logic_str.split()
    tk_arr = [tk for tk in tk_arr if tk != "," and len(tk) > 0]
    tmp_logic = [lg_parent]
    for i in range(len(tk_arr)):
        tk = tk_arr[i]
        if tk == "(":
            if i > 0 and (tk_arr[i - 1] == "(" or tk_arr[i - 1] == ")"):
                new_lg = LogicElement()
                tmp_logic[-1].add_child(new_lg)
                tmp_logic.append(new_lg)
            pass
        elif tk == ")":
            tmp_logic.pop()
        else:
            if i + 1 < len(tk_arr) and tk_arr[i + 1] == "(":
                new_lg = LogicElement(value=tk_arr[i])
                tmp_logic[-1].add_child(new_lg)
                tmp_logic.append(new_lg)
            else:
                tmp_logic[-1].add_child(LogicElement(value=tk))

    return lg_parent


if __name__ == "__main__":
    logic_s = "job ( ANS  ) , job ( ANS  ) , salary_greater_than ( ANS , num_salary , year ) , language ( ANS , languageid0 )"
    logic_s2 = "salary_greater_than ( ANS , num_salary , year ) , job ( ANS ) ,  language ( ANS , languageid0 ) language ( ANS , languageid0 ) salary_greater_than ( ANS , num_salary  ) "
    # logic_s = "( call SW.listValue ( call SW.getProperty ( ( lambda s ( call SW.superlative ( var s ) ( string max ) ( call SW.ensureNumericProperty ( string num_rebounds ) ) ) ) ( call SW.domain ( string player ) ) ) ( string player ) ) )"
    # logic_s = "( lambda ?x exist ?y ( and ( mso:@@ wine.@@ wine.@@ gra@@ pe_@@ vari@@ ety 2005 _ joseph _ car@@ r _ nap@@ a _ valley _ ca@@ ber@@ net _ s@@ au@@ vi@@ gn@@ on ?y ) ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composi@@ tion.@@ gra@@ pe_@@ vari@@ ety ?y pe@@ ti@@ t _ ver@@ do@@ t ) ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composition@@ .per@@ cent@@ age ?y ?x ) ) )"
    logic_s2 = "( lambda ?x exist ?y ( and ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composi@@ tion.@@ gra@@ pe_@@ vari@@ ety ?y pe@@ ti@@ t _ ver@@ do@@ t ) ( mso:@@ wine.@@ wine.@@ gra@@ pe_@@ vari@@ ety 2005 _ joseph _ car@@ r _ nap@@ a _ valley _ ca@@ ber@@ net _ s@@ au@@ vi@@ gn@@ on ?y ) ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composition@@ .per@@ cent@@ age ?y ?x ) ) )"
    # logic_s = "( count $0 ( and ( state:t $0 ) ( exists $1 ( and ( place:t $1 ) ( loc:t $1 $0 ) ( > ( elevation:i $1 ) ( elevation:i ( argmax $2 ( and ( place:t $2 ) ( exists $3 ( and ( loc:t $2 $3 ) ( state:t $3 ) ( loc:t $3 co0 ) ( loc:t ( argmax $4 ( and ( capital:t $4 ) ( city:t $4 ) ) ( size:i $4 ) ) $3 ( size:i $4 ) ) ) ) ) ( elevation:i $2 ) ) ) ) ) ) ) )"
    logic_s2 = "( count $0 ( and ( state:t $0 ) ( exists $1 ( and  ( loc:t $1 $0 ) ( place:t $1 ) ( > ( elevation:i $1 ) ( elevation:i ( argmax $2 ( and ( place:t $2 ) ( exists $3 ( and ( loc:t $2 $3 ) ( state:t $3 ) ( loc:t $3 co0 ) ( loc:t ( argmax $4 ( and ( capital:t $4 ) ( city:t $4 ) ) ( size:i $4 ) ) $3 ( size:i $4 ) ) ) ) ) ( elevation:i $2 ) ) ) ) ) ) ) )"
    logic_s2 = "( ROOT ( S ( SBAR ( WHNP ( ( WDT which ) ) ) ) ( NP ( NNS airline ) ) ( VP ( VBP serve ) ( NP ( NN ci0 ) ) ) ) )"
    # logic_s2 = "( lambda ?x exist ?y ( and ( mso:people.person.marriage meghan_markle ?y ) ( mso:time.event.start_date ?y 9/10/2011 ) ( mso:time.event.location ?y ?x ) ) )"
    # x = function_collapse(logic_s)
    # print(x)
    # print(len(x.split()))
    # print(len(logic_s.split()))
    # print(parse_lambda(logic_s) == parse_lambda(logic_s2))
    # print(parse_lambda(logic_s))
    s2 = parse_lambda(logic_s2)
    print(s2)
    # print(s2)
    leaf = s2.get_leaf_nodes()
    # print(s2.get_triple_name())
    # print(s2.get_constant())
    print(s2.get_path_to_leaf_nodes())

    # print(s2.to_amr())
    # for l in leaf:
    #     if l.is_variable_node():
    #         print(l.to_amr({}))
    #     else:
    #         print("--", l.to_amr({}))
    #         # if l.is_triple():
    #         #     print("xx", l.to_amr())
    #         #     for e in l.child:
    #         #         if e.is_constant():
    #         #             print("++", e.to_amr())
    # print(len(leaf))
    # x = LogicElement(5, sort_child=False)
    # z = LogicElement('baaa')
    #
    # x.add_child(z)
    # x.add_child(LogicElement('aaaa'))
    # print(len(z.child_e))
    #
    # print(x)
# def logic_collapse(logic_str):
#     tokens = logic_str.split()
#     for
