from model.mutation.mutable_base import MutableBase, MutationStrategies, SelectionStrategies
import getopt, sys, os


def build_models(metamodel,base='.',dataset="",nb_base_products=100,training_epochs=25, mutation_rate=0,survival_rate=0, breed=0, evolution_epochs=0, model=""):
    FullEvolution.run(base, last_pdts_path=metamodel, dataset=dataset, nb_base_products=int(nb_base_products), training_epochs=training_epochs, mutation_rate=mutation_rate,survival_rate=survival_rate, breed=breed, evolution_epochs=evolution_epochs, model=model)

def build_meta_model(base=".", input_file="",dataset="mnist",products_file="",pledge_duration=60, training_epochs=25, nb_base_products=(5,5,100)):
    from pledge_evolution import PledgeEvolution, run_pledge
    
    _nb_blocks,_nb_cells, _nb_products = nb_base_products
    output_file = '{}/products_{}s_{}.pdt'.format(base, pledge_duration, "_".join([str(e) for e in nb_base_products]))

    if not os.path.isfile(output_file):
        full_fm_file = PledgeEvolution.end2end(base, nb_base_products, input_file, output_file,
        last_pdts_path=products_file, dataset=dataset, training_epochs=training_epochs )

        run_pledge(full_fm_file, _nb_products, output_file, pledge_path=PledgeEvolution.pledge_path,duration=pledge_duration)
    
    return output_file

def main(argv):

    MutableBase.MAX_NB_CELLS = 5
    MutableBase.MAX_NB_BLOCKS = 10

    MutableBase.selection_strategy = SelectionStrategies.PARETO
    MutableBase.mutation_stategy = MutationStrategies.CHOICE


    input_file = './main_1block_nas.xml'
    output_file = ''
    products_file = ''
    base = './products'
    nb_base_products=[10,5,100]
    #nb_base_products=[int(MutableBase.MAX_NB_BLOCKS/2),MutableBase.MAX_NB_CELLS,100]
    dataset = "mnist"
    model = ""
    training_epochs = 5
    mutation_rate = 0.1
    survival_rate = 0.2
    breed = True
    evolution_epochs = 0
    pledge_duration = 30
    pledge_path = './PLEDGE/PLEDGE.jar'

    opts = []
    try:
        opts, args = getopt.getopt(argv, "hn:t:b:f:a:i:p:d:m:g:r:s:e:y:l:", [
                                   "nb=", "training_epoch=", "bpath=", "fpath=", "ppath=", "dtime=","pfile=", "dataset=","mutation_strategy=","selection_strategy=", "mutation_rate=", "survival_rate=", "evolution_epoch=","breed=","model="])
        
    except getopt.GetoptError:
        print("no arguments")
        pass

    print("arguments {}".format(opts))
    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'run.py --nb <nb architectures BLOCKxCELLxMODELS (10x5x100)> --training_epoch <Number of training epochs (5)> --dtime <Diversity sampling time in s (30)>\
                [--bpath <Base path (./products)> --fpath <Feature Meta-model path (./main_1block_nas.xml)> --ppath <PLEDGE path (./PLEDGE.jar)> --pfile <Products file path> --dataset <Dataset (mnist)>]  \
                [--mutation_strategy <Mutation strategy (random)> --selection_strategy <Selection strategy (pareto)> --mutation_rate <Mutation rate (0.1)> \
                    --survival_rate <Population survival rate (0.2)> --evolution_epoch <Number evolution epochs after diversity sampling (0)> --breed <Breed products after mutation (1)> --model <Base Model ()>]')
            sys.exit()
        elif opt in ("-n", "--nb"):
            nb_base_products = arg.split("x")
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-b", "--bpath"):
            base = arg
        elif opt in ("-f", "--fpath"):
            base = arg
        elif opt in ("-a", "--ppath"):
            pledge_path = arg
        elif opt in ("-i", "--dtime"):
            pledge_duration = arg
        elif opt in ("-p", "--pfile"):
            products_file = arg
        elif opt in ("-t", "--training_epoch"):
            training_epochs = int(arg)
        elif opt in ("-m", "--mutation_strategy"):
            if arg=="all":
                MutableBase.mutation_stategy = MutationStrategies.ALL
            elif arg=="random":
                MutableBase.mutation_stategy = MutationStrategies.CHOICE
        elif opt in ("-g", "--selection_strategy"):     
            if arg =="pareto":
                MutableBase.selection_strategy = SelectionStrategies.PARETO
            elif arg =="elitist":
                MutableBase.selection_strategy = SelectionStrategies.ELITIST
            elif arg =="hybrid":
                MutableBase.selection_strategy = SelectionStrategies.HYBRID

        elif opt in ("-r", "--mutation_rate"):
            mutation_rate = float(arg)
        elif opt in ("-s", "--survival_rate"):
            survival_rate = float(arg)
        elif opt in ("-e", "--evolution_epoch"):
            evolution_epochs = int(arg)
        elif opt in ("-y", "--breed"):
            breed = bool(arg)
        elif opt in ("-l", "--model"):
            model = arg
        

    from full_evolution import FullEvolution
    from pledge_evolution import PledgeEvolution, run_pledge
        
    _nb_blocks,_nb_cells, _nb_products = nb_base_products    
    PledgeEvolution.pledge_path = pledge_path
    
    if len(nb_base_products) ==1:
        FullEvolution.run(base, last_pdts_path=products_file, dataset=dataset, nb_base_products=int(nb_base_products[0]), training_epochs=training_epochs, mutation_rate=mutation_rate,survival_rate=survival_rate, breed=breed, evolution_epochs=evolution_epochs)
    else:
        metamodel = build_meta_model(base=base, input_file=input_file,dataset=dataset,products_file=products_file,pledge_duration=pledge_duration, training_epochs=training_epochs, nb_base_products=(5,5,100))

        build_models(metamodel=metamodel, base=base, dataset=dataset, nb_base_products=int(_nb_products), training_epochs=training_epochs, mutation_rate=mutation_rate,survival_rate=survival_rate, breed=breed, evolution_epochs=evolution_epochs, model=model)


if __name__ == "__main__":
    print("Running FeatureNET model sampler")
    main(sys.argv[1:])
