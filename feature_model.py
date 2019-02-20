## https://github.com/arennax/fss18_xia

from __future__ import division, print_function
from FeatureModel.Feature_tree import FeatureTree
import random
import pdb

"""
This module handles the transactions of feature model. All feature model  are formatted as SPLOT xml file,
such as http://52.32.1.180:8080/SPLOT/models/model_20170930_228024571.xml
"""

if __name__ == '__main__':
    # Step 1. Load SPLIT model. Use load_ft_from_url()
    # here url can be www url OR file path

    # url = "http://52.32.1.180:8080/SPLOT/models/model_20170930_228024571.xml"
    url = "./FeatureModel/tree_model.xml"
    ft = FeatureTree()
    ft.load_ft_from_url(url)

    # Step 2. Check feature model information
    print('This model has\n'
          ' {0} constraints\n'
          ' {1} features\n'
          ' {2} depth'.format(
        ft.get_cons_num(),
        ft.get_feature_num(),
        ft.get_tree_height()
    ))

    # Step 3. generate the configurations
    given = {f: random.choice([0, 1]) for f in ft.leaves}
    X = ft.get_full_feature_configure_by_partial_def(given, dict)
    pdb.set_trace()
    print(ft.leaves[0].name)
    # Step 4. checking whether our configuration is correct
    isvalid = ft.check_fulfill_valid(X)
    print("Valid configure? %s" % isvalid)
