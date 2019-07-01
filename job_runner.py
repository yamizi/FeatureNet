from model.mutation.mutable_base import MutableBase, MutationStrategies, SelectionStrategies
import getopt, sys, os

def main(argv):

    MutableBase.MAX_NB_CELLS = 5
    MutableBase.MAX_NB_BLOCKS = 10

    MutableBase.selection_stragey = SelectionStrategies.PARETO
    MutableBase.mutation_stategy = MutationStrategies.CHOICE


    input_file = ''
    output_file = ''
    products_file = ''
    base = '../products'
    nb_base_products=[20]
    nb_base_products=[MutableBase.MAX_NB_BLOCKS,MutableBase.MAX_NB_CELLS,20]
    dataset = "cifar"
    training_epochs = 12
    mutation_rate = 0.1
    survival_rate = 0.1
    breed = True
    evolution_epochs = 50

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
    from pledge_evolution import PledgeEvolution, run_pledge
        
    if len(nb_base_products) ==1:
        FullEvolution.run(base, last_pdts_path=products_file, dataset=dataset, nb_base_products=nb_base_products, training_epochs=training_epochs, mutation_rate=mutation_rate,survival_rate=survival_rate, breed=breed, evolution_epochs=evolution_epochs)
    else:
        _nb_blocks,_nb_cells, _nb_products = nb_base_products
        
        full_fm_file = PledgeEvolution.end2end(base, nb_base_products, input_file, output_file,
        last_pdts_path=products_file, dataset=dataset, training_epochs=training_epochs )

        output_file = '{}/products_{}.pdt'.format(base, "_".join([str(e) for e in nb_base_products]))

        if not os.path.isfile(output_file):
            run_pledge(full_fm_file, _nb_products, output_file, duration=300)

        FullEvolution.run(base, last_pdts_path=output_file, dataset=dataset, nb_base_products=_nb_products, training_epochs=training_epochs, mutation_rate=mutation_rate,survival_rate=survival_rate, breed=breed, evolution_epochs=evolution_epochs)
    

if __name__ == "__main__":
    main(sys.argv[1:])
