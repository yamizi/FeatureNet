class Node(object):
    def __init__(self, raw_dict=None):
            self.raw_dict = raw_dict
            print(self.get_name())

    def get_name(self):
        if self.raw_dict:
            return self.raw_dict.get("label")
        return ""

    @staticmethod
    def get_type(element, keep_index=True):
        element_type = element.get("label")
        element_type = element_type[element_type.rfind("_")+1:]

        if not keep_index:
            element_type = ''.join([i for i in element_type if not i.isdigit()])

        return element_type.lower()