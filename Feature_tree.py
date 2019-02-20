import random
import re
import sys
import time
import urllib
import xml.etree.ElementTree
import time
import pdb

import validators

class Node(object):
    def __init__(self, identification, name, parent=None, node_type='o'):
        self.id = identification
        self.parent = parent
        self.node_type = node_type
        self.children = []
        self.name = name.strip().replace(' ', '_')
        if node_type == 'g':
            self.g_u = 1
            self.g_d = 0

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

    def __repr__(self):
        return '%s|%s %s' % (self.node_type, self.id, self.name)


class Constraint(object):
    def __init__(self, identification, literals, literals_pos):
        self.id = identification
        self.literals = literals
        self.li_pos = literals_pos

    def __repr__(self):
        return self.id + '\n' + str(self.literals) + '\n' + str(self.li_pos)

    def is_correct(self, ft, filled_form):
        """
        not supported for filled_form containing -1
        try to apply is_violate if needed.
        """
        for li, pos in zip(self.literals, self.li_pos):
            i = ft.find_fea_index(li)
            if int(pos) == filled_form[i]:
                return True
        return False

    def is_violated(self, ft, filled_form):
        for li, pos in zip(self.literals, self.li_pos):
            i = ft.find_fea_index(li)
            filled = filled_form[i]
            if filled == -1 or int(pos) == filled_form[i]:
                return False
        return True


class FeatureTree(object):
    def __init__(self):
        self.fea_index_dict = dict()
        self.root = None
        self.features = []
        self.groups = []
        self.leaves = []
        self.con = []
        self.featureNum = 0
        self.subtree_index_dict = dict()

    def _set_root(self, root):
        self.root = root

    def _add_constraint(self, con):
        self.con.append(con)

    def find_fea_index(self, id_or_nodeObj):
        if type(id_or_nodeObj) is not str:
            identification = id_or_nodeObj.id
        else:
            identification = id_or_nodeObj

        for f_i, f in enumerate(self.features):
            self.fea_index_dict[f.id] = f_i

        return self.fea_index_dict[identification]

    # fetch all the features in the tree basing on the children structure
    def set_features_list(self):
        def setting_feature_list(node):
            if node.node_type == 'g':
                node.g_u = int(node.g_u) if node.g_u != sys.maxsize else len(node.children)
                node.g_d = int(node.g_d) if node.g_d != sys.maxsize else len(node.children)
                self.features.append(node)
                self.groups.append(node)
            if node.node_type != 'g':
                self.features.append(node)
            if len(node.children) == 0:
                self.leaves.append(node)
            for i in node.children:
                setting_feature_list(i)

        setting_feature_list(self.root)
        self.featureNum = len(self.features)

    def post_order(self, node, func, extra_args=None):
        """children, then the root"""
        if extra_args is None:
            extra_args = []
        if node.children:
            for c in node.children:
                self.post_order(c, func, extra_args)
        func(node, *extra_args)

    def pre_order(self, node, func, extra_args=None):
        """root, then children"""
        if extra_args is None:
            extra_args = []
        func(node, *extra_args)
        if node.children:
            for c in node.children:
                self.pre_order(c, func, extra_args)

    def check_fulfill_valid(self, fill):
        """
        checking a given fulfill lst whether consistent with the feature model tree structure
        :param fill:
        :return:
        """
        if type(fill) == dict:
            tmp = [fill[feature] for feature in self.features]
            fill = tmp

        def find(x):
            return fill[self.find_fea_index(x)]

        def check_node(node):
            if not node.children:
                return True

            if find(node) == 0:
                return True

            child_sum = sum([find(c) for c in node.children])

            for m_child in filter(lambda x: x.node_type in ['m', 'r', 'g'], node.children):
                if find(m_child) == 0:
                    return False

            if node.node_type is 'g':
                # pdb.set_trace()
                if not (node.g_d <= child_sum <= node.g_u):
                    return False

            for child in node.children:
                if find(child) == 1:
                    t = check_node(child)
                    if not t:
                        return False
            return True

        if fill[0] == 0:
            return False
        return check_node(self.root)

    def get_feature_num(self):
        return len(self.features) - len(self.groups)

    def get_cons_num(self):
        return len(self.con)

    def get_tree_height(self):
        h_dict = dict()

        def inner(f):
            if f.node_type == "" or not f.children:
                h_dict[f] = 1
            else:
                h_dict[f] = max([h_dict[x] for x in f.children]) + 1

        self.post_order(self.root, inner)

        return h_dict[self.root]

    def load_ft_from_url(self, url):
        if validators.url(url):
            url = urllib.urlopen(url)
        # load the feature tree and constraints
        tree = xml.etree.ElementTree.parse(url)
        root = tree.getroot()
        

        for child in root:
            if child.tag == 'feature_tree':
                feature_tree = child.text
            if child.tag == 'constraints':
                constraints = child.text
        # parse the feature tree text
        features = feature_tree.split("\n")
        features = filter(bool, features)
        common_feature_pattern = re.compile('(\t*):([romg]?)(.*)\W(\w+)\W.*')
        group_pattern = re.compile('\t*:g(\w*) \W(\d),([\d\*])\W.*')
        layer_dict = dict()
        for f in features:
            m = common_feature_pattern.match(f)
            """
            m.group(1) layer
            m.group(2) type
            m.group(3) name
            m.group(4) id
            """
            print(m.group(0))
            layer = len(m.group(1))
            t = m.group(2)
            if t == 'r':
                tree_root = Node(identification=m.group(4), name=m.group(3), node_type='r')
                layer_dict[layer] = tree_root
                self._set_root(tree_root)
            elif t == 'g':
                mg = group_pattern.match(f)
                
                """
                mg.group(1) id
                mg.group(2) down_count
                mg.group(3) up_count
                """
                identification_ = mg.group(1)
                if not identification_:
                    identification_ = "{0}_".format(layer_dict[layer - 1].id) 
                gNode = Node(identification=identification_ , name='g', parent=layer_dict[layer - 1], node_type='g')
                layer_dict[layer] = gNode
                if mg.group(3) == '*':
                    gNode.g_u = sys.maxsize
                else:
                    gNode.g_u = mg.group(3)
                gNode.g_d = mg.group(2)
                layer_dict[layer] = gNode
                gNode.parent.add_child(gNode)
            else:
                treeNode = Node(identification=m.group(4), name=m.group(3), parent=layer_dict[layer - 1], node_type=t)
                layer_dict[layer] = treeNode
                treeNode.parent.add_child(treeNode)

        # parse the constraints
        cons = constraints.split('\n')
        cons = filter(bool, cons)
        common_con_pattern = re.compile('(\w+):(~?)(\w+)(.*)\s*')
        common_more_con_pattern = re.compile('\s+(or) (~?)(\w+)(.*)\s*')

        for cc in cons:
            literal = []
            li_pos = []
            m = common_con_pattern.match(cc)
            con_id = m.group(1)
            li_pos.append(not bool(m.group(2)))
            literal.append(m.group(3))
            while m.group(4):
                cc = m.group(4)
                m = common_more_con_pattern.match(cc)
                li_pos.append(not bool(m.group(2)))
                literal.append(m.group(3))
            """
             con_id: constraint identifier
             literal: literals
             li_pos: whether is positive or each literals
            """
            con_stmt = Constraint(identification=con_id, literals=literal, literals_pos=li_pos)
            self._add_constraint(con_stmt)

        self.set_features_list()

    def get_full_feature_configure_by_partial_def(self, given, type_of_return=list):
        """
        for all unassigned features, we use random_config assignment.
        DO NOT GRANTEE TO BE VALID AFTER GENERATION
        :param given: typically assignments of all leaves
        :param type_of_return: setting which type to return. can be list or dict
        :return:
        """
        configure = {k: v for k, v in given.items()}
        startat = time.time()
        while len(configure) < len(self.features):
            for feature in self.features:
                if feature in configure:
                    continue
                if feature.node_type == '':  # the leave
                    configure[feature] = random.choice([0, 1])
                elif feature.node_type == 'm' or feature.node_type == 'r':
                    configure[feature] = 1
                elif feature.node_type == 'o':
                    tmp = [configure[s] for s in feature.children if s in configure]
                    if 1 in tmp:
                        configure[feature] = 1
                    elif len(tmp) == len(feature.children):
                        configure[feature] = 0
                elif feature.node_type == 'g':
                    tmp = [configure[s] for s in feature.children if s in configure]
                    if len(tmp) == len(feature.children):
                        configure[feature] = 1 if feature.g_d <= sum(tmp) <= feature.g_u else 0

            assert time.time() - startat < 15000, "please check this loop. not end"

        if type_of_return == dict:
            return configure
        elif type_of_return == list:
            con_lst = []
            for feature in self.features:
                con_lst.append(configure[feature])
            return con_lst
        else:
            sys.stderr.wirte("check type_of_return input")
            return configure

    def top_down_random(self, random_seed=None):
        """
        Use top-down strategy to randomly generate a configuration
        Return is NOT necessary to be valid,
        but it is more likely to be valid than gen_full_feature_configure_by_partial_def(), since it takes
            tree structure and group limits into consideration
        :return:
        """
        configure = {k: -1 for k in self.features}

        # set root first
        configure[self.root] = 1

        def fill_child(node):
            if configure[node] == -1:
                assert False, "check here"

            if configure[node] == 0:
                for c in node.children:
                    configure[c] = 0
                return

            if node.node_type == 'g':  # grantee the group limits
                samples = random.sample(node.children, random.randint(node.g_d, node.g_u))
                for c in node.children:
                    configure[c] = 1 if c in samples else 0
                return

            for c in node.children:
                if c.node_type == 'r' or c.node_type == 'm' or c.node_type == 'g':
                    configure[c] = 1
                else:
                    configure[c] = random.choice([0, 1])
            return

        self.pre_order(self.root, fill_child)

        return configure