from model.mutation.mutable_base import MutableBase, MutationStrategies, SelectionStrategies
import getopt, sys

def main(argv):
    input_file = ''
    output_file = ''
    products_file = ''
    base = '../products'
    nb_base_products=[20]
    dataset = "cifar"
    training_epochs = 12
    mutation_rate = 0.1
    survival_rate = 0.1
    breed = True
    evolution_epochs = 50

    MutableBase.MAX_NB_CELLS = 5
    MutableBase.MAX_NB_BLOCKS = 10

    MutableBase.selection_stragey = SelectionStrategies.HYBRID
    MutableBase.mutation_stategy = MutationStrategies.CHOICE

    opts = []
    try:
        opts, args = getopt.getopt(argv, "hn:d:b:p:t:m:r:s:e:", [
                                   "nb=","dataset=", "bpath=", "pfile=", "training_epoch=", "mutation_strategy=", "mutation_rate=", "survival_rate=", "evolution_epoch="])
        
    except getopt.GetoptError:
        print("no arguments")
        pass

    print("arguments {}".format(opts))
    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'pledge_evolution.py -n <nb_architectures> -d <dataset> -b <base_path> -p <products_file> -t <training_epoch> -m <mutation_strategy> -r <mutation_rate> -s <survival_rate>')
            sys.exit()
        elif opt in ("-n", "--nb"):
            nb_base_products = arg.split("x")
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-b", "--bpath"):
            base = arg
        elif opt in ("-p", "--pfile"):
            products_file = arg
        elif opt in ("-t", "--training_epoch"):
            training_epochs = int(arg)
        elif opt in ("-m", "--mutation_strategy"):
            args = arg.split("#")
            if args[0]=="all":
                MutableBase.mutation_stategy = MutationStrategies.ALL
            if len(args)>1 and args[1]=="pareto":
                MutableBase.selection_stragey = SelectionStrategies.PARETO
        elif opt in ("-r", "--mutation_rate"):
            mutation_rate = float(arg)
        elif opt in ("-s", "--survival_rate"):
            survival_rate = float(arg)
        elif opt in ("-e", "--evolution_epoch"):
            evolution_epochs = int(arg)

    from full_evolution import FullEvolution
    FullEvolution.run(base, last_pdts_path=products_file, dataset=dataset, nb_base_products=int(nb_base_products[0]), training_epochs=training_epochs, mutation_rate=mutation_rate,survival_rate=survival_rate, breed=breed, evolution_epochs=evolution_epochs)

if __name__ == "__main__":
    main(sys.argv[1:])
