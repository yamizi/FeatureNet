from __future__ import division, print_function
from feature_tree import FeatureTree
from products_tree import ProductSet
import random
import pdb


def loadFeatures():
  # Step 1. Load SPLIT model. Use load_ft_from_url()

    url = "C:/Users/salah.ghamizi/Downloads/pledge-master/tensorflow//NAS Model.xml"
    ft = FeatureTree()
    ft.load_ft_from_url(url)

    # Step 2. Check feature model information
    print('This model has/n'
          ' {0} constraints/n'
          ' {1} features/n'
          ' {2} depth'.format(
        ft.get_cons_num(),
        ft.get_feature_num(),
        ft.get_tree_height()
    ))

def loadProducts():
  url = "C:/Users/salah.ghamizi/Downloads/pledge-master/tensorflow//NAS Products.pdt"
  productSet = ProductSet(url)
  print("number of features {0}, number of products: {1}".format(productSet.nbFeatures, productSet.nbProducts))

  rand_product = productSet.format_product(2)#random.randrange(0, productSet.nbProducts-1))
  print((rand_product))

  products = productSet.format_products()
  for i, prd in enumerate(products):
    if len(prd) < 10:
      print("{0}:{1}".format(i, len(prd)))
  
if __name__ == '__main__':
  loadProducts()