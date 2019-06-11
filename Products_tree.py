import re
import sys, os
import validators


class ProductSetError(Exception):
    pass
    

class ProductSet(object):


    @staticmethod
    def filter_leaves(features, keep_list=None):
        if not keep_list:
            keep_list = ("_features_", "_activation_", "_kernel_", "_type_","_stride_","_padding_", "_Dropout_value_", "_BatchNormalization","_Void", "relativeCellIndex_", "Output_Block","_Combination_")
            filtered = {k:features[k] for k in features.keys() if any(( features[k].find(keep) !=-1 for keep in keep_list))}

        #features = sorted(features.items(), key = lambda kv:(kv[1], kv[0]))
        filtered = {k:features[k] for k in features.keys() if len(k)>3}
        
        return filtered


    def __init__(self, url, binary_products = False):
        self.features = {}
        self.constraints = []
        self.products = []
        self.nbFeatures = 0;
        self.nbProducts = 0;
        self.last_products_url = "";
        self.binary_products = binary_products

        if url:
            self.load_products_from_url(url)

    def load_products_from_url(self, url):
    
        print("==> Loading products from {}".format(url))

        if not os.path.isfile(url):
            print("File not found")
            raise ProductSetError()

        self.last_products_url = url 
        self.features = {}
        self._features_reverse = {}
        self.constraints = []
        self.products = []

        feature_pattern = re.compile('(\d*)->(\w*)')
        f = open(url, 'r')
        line = f.readline()
        while line:
            #print(line)
            
            feature = feature_pattern.match(line)
            if feature:
                    self.features[feature.group(1)] = feature.group(2)
                    self._features_reverse[feature.group(2)] = feature.group(1)
            else:

                product_features = line.split(";")
                product_features = product_features[:-1]
                if self.binary_products:
                    product_features = sorted( product_features, key=lambda k: abs(int(k)))
                    product_features =  [1 if str(x).isdigit() and int(x)>0 else 0 for x in product_features]

                self.products.append(product_features) 
            line = f.readline()
        f.close()
        self.nbFeatures = len(self.features.keys())
        self.nbProducts = len(self.products)

    def format_product(self,prd_index=0, original_product=None, include_original=True, sort_features=False):
        #product = [self.features.get(str(x)) for x in self.products[index]]
        # to dict
        
        original_product =  self.products[prd_index] if not original_product else original_product
        if not original_product:
            return None

        if self.binary_products:
            nb_features = len(original_product)
            product = [(i+1) for i in range(nb_features) if  int(original_product[i])>0]
        else:
            product = [abs(int(x)) for i,x in enumerate(original_product) if  str(x).isdigit() and int(x)>=0]

    
        product_labels = {self.features[str(x)]:i for i,x in enumerate(product)}
        product_nodes = [{'label':self.features[str(x)],'id':x, "children":[]} for x in product]

        #reversed list
        for i in product[::-1]:
            label= self.features.get(str(i))
            parent_label = label[0:label.rfind("_")] if label.rfind("_") > -1 else ""

            if parent_label:
                parent_product_index =product_labels.get(parent_label)
                if parent_product_index:
                    product_nodes[parent_product_index].get("children").append(product_nodes[product.index(i)])

        #only return nodes that represents the blocks
        prod =  [nodes for nodes in product_nodes if nodes.get("label").startswith("Block") and nodes.get("label").find("_")==-1]

        if include_original:
            sorted_features = sorted( original_product, key=lambda k: abs(int(k))) if sort_features else original_product
            return prod, sorted_features
        else:
            return prod
    

    def light_product(self, prd_index=0, product=None):
        if not product:
            product =  self.products[prd_index]

        product_nodes, features = self.format_product(original_product=product)
        
        def light_label(node):
            node["label"] =  node["label"][node["label"].rfind("_"):]
            node["children"] = sorted( [light_label(child) for child in node["children"]], key=lambda k: k['id'])
            return node

        export_product = [light_label(prd) for prd in product_nodes]
        return export_product


    
    def format_products(self, include_original=True, sort_features=False):
        prds = []
        for product in self.products:
            prds.append(self.format_product(original_product=product, include_original=include_original,sort_features=sort_features))
            
        return prds

    def light_products(self):
        for product in self.products:
            yield self.light_product(product=product)
