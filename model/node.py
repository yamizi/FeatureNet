import uuid
class Node(object):
    content = None
    parent = None
    parent_model = None
    layer_mapping = {}
    node_mapping = {}
    node_list = []
    increment = 0
     
    def __init__(self, raw_dict=None, parent_model=None):
        self.parent_name = ""
        self.raw_dict = raw_dict
        self.customizable_parameters = {}
        self.uniqid = "{}-{}".format(Node.increment,str(uuid.uuid1())[:10])
        Node.increment = Node.increment+1
        Node.node_mapping[self.name] = self
        Node.node_list.append(self.name)

        if parent_model:
            self.parent_model = parent_model

        #print(self.get_name())

    def __deepcopy__(self, memo):
        from copy import deepcopy

        uniqid = "{}_{}-{}".format(self.uniqid, Node.increment,str(uuid.uuid1())[:10])
        Node.increment = Node.increment + 1

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k =="last_build":
                continue
            elif k =="uniqid":

                setattr(result, "uniqid", deepcopy(uniqid, memo))
            else:
                setattr(result, k, deepcopy(v, memo))

        Node.node_mapping[result.name] = result
        Node.node_list.append(result.name)

        return result

    def get_name(self, raw_dict=None):
        if self.raw_dict:
            lbl = self.raw_dict.get("label")
            lbl = lbl.replace("Block","-B-").replace("Cell_","").replace("Element","-C-")
            return self.uniqid+lbl

        return self.uniqid

    def append_parameter(self, attribute, possible_values=""):
        self.customizable_parameters[attribute] = possible_values

    def get_custom_parameters(self):
        my_params = self.customizable_parameters
        params = {}
        if len(my_params.keys()):
            params = {self.get_name():(self, my_params)}

        return params

    @property
    def name(self):
        return self.get_name()+ "-n-" +self.__class__.__name__

    @property
    def fullname(self):
        classname =  self.__class__.__name__
        node_name =self.get_name()
        return  node_name + "-f-" +classname

        
    @staticmethod
    def layer_to_cell(layer):
        node = Node.layer_to_node(layer)
        cell = node.parent_cell
        return cell
        
    @staticmethod
    def layer_to_node(layer):
        node = Node.layer_mapping.get(layer.name)
        return Node.node_mapping.get(node)

    @staticmethod
    def node_to_layer(node):
        layers = [l for (l,n) in Node.layer_mapping.items() if n==node.name]
        return layers[0]

    @staticmethod
    def get_type(element, keep_index=True):
        element_type = element.get("label")
        element_type = element_type[element_type.rfind("_")+1:]

        if not keep_index:
            element_type = ''.join([i for i in element_type if not i.isdigit()])

        return element_type.lower()

    @staticmethod
    def reset_nodes():

        Node.content = None
        Node.parent = None
        Node.parent_model = None
        Node.layer_mapping = {}
        Node.node_mapping = {}
        Node.node_list = []
        Node.increment = 0

    @property
    def parent_cell(self):
        return self._parent_cell

    @parent_cell.setter
    def parent_cell(self, value):
        self._parent_cell = value
        if value:
            self.parent_model = value.parent_model
