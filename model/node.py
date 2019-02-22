class Node(object):
    def __init__(self, raw_dict=None):
            self.raw_dict = raw_dict
            self.customizable_parameters = {}
            print(self.get_name())

    def get_name(self, raw_dict=None):
        if self.raw_dict:
            return self.raw_dict.get("label")
        return ""

    def append_parameter(self, attribute, possible_values=""):
        self.customizable_parameters[attribute] = possible_values

    def get_custom_parameters(self):
        my_params = self.customizable_parameters
        params = {}
        if len(my_params.keys()):
            params = {self.get_name():(self, my_params)}

        return params

    @staticmethod
    def get_type(element, keep_index=True):
        element_type = element.get("label")
        element_type = element_type[element_type.rfind("_")+1:]

        if not keep_index:
            element_type = ''.join([i for i in element_type if not i.isdigit()])

        return element_type.lower()