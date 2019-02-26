from products_tree import ProductSet
import json

def parse_products():
   
    url = "C:/Users/salah.ghamizi/Documents/PhD/spl/tensorflow/10Products.pdt"
    productSet = ProductSet(url)
    print("number of features {0}, number of products: {1}".format(productSet.nbFeatures, productSet.nbProducts))    
    product = productSet.format_product(2)

    f = open("./block2.json", "w")
    f.write(json.dumps([product[0]])) 

parse_products()