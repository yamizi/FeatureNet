import re
import sys
import validators

class ProductSet(object):
    def __init__(self, url):
        self.features = {}
        self.constraints = []
        self.products = []
        self.nbFeatures = 0;
        self.nbProducts = 0;
        if url:
            self.load_products_from_url(url)

    def load_products_from_url(self, url):
    
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

                product_features = [abs(int(x)) for x in line.split(";") if  x.isdigit() and int(x)>0]
                self.products.append(product_features) 
            line = f.readline()
        f.close()
        self.nbFeatures = len(self.features.keys())
        self.nbProducts = len(self.products)

    def format_product(self,prd_index=0, product=None):
        #product = [self.features.get(str(x)) for x in self.products[index]]
        # to dict

        if not product:
            product =  self.products[prd_index]
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
        return [nodes for nodes in product_nodes if nodes.get("label").startswith("Block") and nodes.get("label").find("_")==-1]
    

    def light_product(self, prd_index=0, product=None):
        if not product:
            product =  self.products[prd_index]

        product_nodes = self.format_product(product=product)
        
        def light_label(node):
            node["label"] =  node["label"][node["label"].rfind("_"):]
            node["children"] = sorted( [light_label(child) for child in node["children"]], key=lambda k: k['id'])
            return node

        export_product = [light_label(prd) for prd in product_nodes]
        return export_product


    
    def format_products(self):
        prds = []
        for product in self.products:
            prds.append(self.format_product(product=product))
            
        return prds

    def light_products(self):
        for product in self.products:
            yield self.light_product(product=product)