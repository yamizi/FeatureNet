import subprocess
import os

print(os.getcwd())
print(os.path.dirname(os.path.abspath(__file__)))

nb_products = 1000
base_path = '../products/'
inputfile = base_path+'main_5_5.xml'  
outputfile = base_path+'main_5_5.pdt'  

params = ['java', '-jar', base_path+'PLEDGE.jar','generate_products']
params = params+['-fm',inputfile,'-nbProds',str(nb_products),'-o',outputfile]
subprocess.call(params)
