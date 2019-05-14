
class MutableBase(object):

    def mutate_attributes(self):
        pass
    
    def mutate_child(self):
        pass

    def duplicate(self):
        pass

    def remove(self):
        pass
    
    def mutate(self):
        pass

    def __init__(self, raw_dict=None, previous_block = None):

        super(MutableBase, self).__init__(raw_dict,previous_block)
