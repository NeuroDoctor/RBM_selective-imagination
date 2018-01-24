from RBM_class_BATCH import RBM
import datetime
import numpy as np


Set_fixed_casuality = np.random.seed(3) # fix the casuality to debug in a better manner

# LOADING IMAGES SIZE AND INPUTS UNITS DEPENDING ON IMAGES SIZE

Images_dimensions = 1 #input (" original dimension (0) or resize dimensions (1) ? \n ")

MNIST = 0 #input (" do yo want to load MNIST ? \n ")


#STRUCTURE AND LEARNING / RUNNING PARAMETRERS OF NETWORK

numbers_layers = 2 #input (" How many layers (1 or 2)?   \n ")

#ASSIGNATION OF LAYERS

first = 1 # FIRST LAYER

second = 2 # SECOND LAYER

test_network = 1 #input (" Do you want to test network? (1 for yes or 0 for no)  \n ")

# FOUNDAMENTAL PARAMETRERS OF FIRST LAYER

save_choice = 0  #input (" do you want to save weights of first layer ? \n ")

load_choice = 1 #input (" do you want to load weights of first layer (if you load weights, you have not to do the learning)? \n ")

Hidden_units = 100 #input (" how many hidden units  ? \n ")

plot_choice = 1 #input (" do you want to choose how many hiddens will be plotted ? (1 for yes or 0 for no) \n ")

hiddens_plotted = 10 #input (" how many hiddens will plotted ? \n ")

Batch_size_first = 9  #input (" Batch size of first layer? \n ")

TotEpoc = 3500 #input (" how many epocs for first layer ? \n ")

# FOUNDAMENTAL PARAMETRERS OF SECOND LAYER

if numbers_layers == 2:

    Hidden_units_second = 6 #input (" how many output units ? \n ")

    Hidden_units_second_plotted = 10  # input (" how many outputs will plotted? \n ")


#PROPRIERTIES SELECTION

proprierties = 0 # input ("  What proprierty will be selected for renconstruction (0 no selection, 1 for shape, 2 for color, another number for exagerated selection) ? \n ")



#%%%INITIALIZATION OF FIRST RBM%%%

RBM_obj = RBM(1,(Hidden_units)) #init of first RBM

RBM_obj.get_images(Images_dimensions, MNIST) # loading inputs images

RBM_obj.variables_changes_for_images(first)

RBM_obj.initialization_weights(RBM_obj.input_units,(Hidden_units)) # initialization of weights_ first RBM

if MNIST == 1:

    Total_inputs = RBM_obj.originals_matrices.shape[0] # return the numbers of inputs (MNIST = 60.000 )

elif MNIST == 0:

    Total_inputs = len(RBM_obj.originals_matrices) # return the numbers of inputs (9 polygons returns n = 9)

#%%%LEARNING OF RBM%%%


if load_choice == 0:

    # LEARNING FIRST LAYER

    #starting variables of first layer

    epoc = 0
    time = 0
    batch_single = 0
    Starting_time = datetime.datetime.today()


    TotEpoc = 50000
    RBM_obj.K = 0
    RBM_obj.learning_rate = 0.1
    min_error = 0.2


    print "Start first layer learning..." '\n'

    while epoc < TotEpoc:


        Step_time = datetime.datetime.today()

        print "epoc n ",epoc, '\n'
        print "time step = ", Step_time - Starting_time, '\n'


        while batch_single < Total_inputs / Batch_size_first:

            #print "batch n ",batch


            RBM_obj.get_input(first, MNIST, batch_single, Batch_size_first,0, test_network)
            RBM_obj.get_output_first_layer()
            RBM_obj.input_reconstruction()
            RBM_obj.get_output_first_layer()
            RBM_obj.update_weights_input_hidden()
            RBM_obj.update_BATCH_input_hidden(Batch_size_first)
            batch_single += 1


        #PRINTS AND ERRORS APPENDS

        print " epoc reconstruction error (Inputs) = ", np.around(RBM_obj.percent_error_1, decimals = 5), '\n'

        RBM_obj.FirstErrors.append(RBM_obj.percent_error_1)

        # for i in range(0,Total_inputs):
        #     RBM_obj.single_input_errors[i].append(np.around((((RBM_obj.errors_for_input[i]) * 100)), decimals = 5))


        #COUNTS AND LEARNING CONTROL ( ERR < MINIMUM VALUE)

        batch_single = 0
        epoc += 1

        if epoc > 5:

            RBM_obj.alfa = 0.9

        if RBM_obj.percent_error_1 < min_error:
            break

    print "...Finish first layer learning"' \n'

    RBM_obj.save_weights(save_choice)

elif load_choice == 1:

    RBM_obj.load_weights(load_choice)




#%%%INITIALIZATION OF SECOND RBM%%%

if numbers_layers == 2:

    RBM_obj_second = RBM(2, (Hidden_units_second))  # init of second RBM

    RBM_obj_second.variables_changes_for_images(second, RBM_obj.hidden_output)

    RBM_obj_second.initialization_weights(Hidden_units, (Hidden_units_second))  # initialization of weights_ second RBM


#LEARNING SECOND LAYER

if numbers_layers == 2:



        epoc = 0
        time = 0
        batch_single = 0
        Starting_time = datetime.datetime.today()
        RBM_obj_second.input = RBM_obj.hidden_output


        TotEpoc = 10000
        RBM_obj_second.learning_rate = 0.01
        RBM_obj_second.k = 0
        Batch_size_first = 1
        min_error_second = 5


        print "Start second layer learning..." '\n'


        while epoc < TotEpoc:


            Step_time = datetime.datetime.today()

            print "epoc n ",epoc, '\n'
            print "time step = ", Step_time - Starting_time, '\n'


            while batch_single < Total_inputs / Batch_size_first:

                #print "batch n ",batch


                RBM_obj_second.get_input(second, MNIST, batch_single, Batch_size_first,RBM_obj.hidden_output, test_network)
                RBM_obj_second.get_output_first_layer()

                RBM_obj_second.selective_learning(1,proprierties)

                RBM_obj_second.input_reconstruction()
                RBM_obj_second.get_output_first_layer()
                RBM_obj_second.update_weights_input_hidden()
                RBM_obj_second.update_BATCH_input_hidden(Batch_size_first)
                batch_single += 1


            #PRINTS AND ERRORS APPENDS

            print " epoc reconstruction error (Inputs) = ", np.around(RBM_obj_second.percent_error_1, decimals = 5), '\n'

            RBM_obj_second.FirstErrors.append(RBM_obj_second.percent_error_1)

            # for i in range(0,Total_inputs):
            #     RBM_obj_second.single_input_errors[i].append(np.around((((RBM_obj_second.errors_for_input[i]) * 100)), decimals = 5))


            #COUNTS AND LEARNING CONTROL ( ERR < MINIMUM VALUE)

            batch_single = 0
            epoc += 1

            if epoc > 5:

                RBM_obj_second.alfa = 0.9

            if RBM_obj_second.percent_error_1 < min_error_second:
                break

        print "...Finish second layer learning"' \n'






#%%%TEST OF RBM%%%


# TEST FOR 1 LAYER RBM

numbers_layers = 1

if numbers_layers == 1 and test_network == 1:


        print "start test first layer..." '\n'

        RBM_obj.K = 0

        RBM_obj.get_input(first, MNIST, batch_single, Batch_size_first, test_network)
        RBM_obj.get_output_first_layer(test_network)
        RBM_obj.input_reconstruction(0,1)

        if load_choice == 0:
            RBM_obj.Plot_errors(numbers_layers)
            RBM_obj.Graphical_reconstruction_inputs_hiddens_outputs(RBM_obj, Total_inputs, plot_choice, Hidden_units, hiddens_plotted,numbers_layers)

        print "...finish test first layer" '\n'

# TEST FOR 2 LAYERS RBM

numbers_layers = 2

if numbers_layers == 2 and test_network == 1: # TEST FOR 2 LAYERS RBM

    print "start test second layer..."'\n'

    RBM_obj_second.K = 0

    RBM_obj.get_input(first, MNIST, batch_single, Batch_size_first, test_network)
    RBM_obj.get_output_first_layer(test_network)

    RBM_obj_second.get_input(second, MNIST, batch_single, Batch_size_first,RBM_obj.hidden_output, test_network)
    RBM_obj_second.get_output_first_layer(test_network, second)

    RBM_obj_second.perceptron_selective_reconstruction(proprierties)

    print "reconstruction of first hidden layer"'\n'

    RBM_obj_second.input_reconstruction(0,1)
    RBM_obj.hidden_output = RBM_obj_second.input

    print "reconstruction of input layer"'\n'

    RBM_obj.input_reconstruction(0,1)

    RBM_obj_second.Plot_errors(numbers_layers)
    RBM_obj_second.Graphical_reconstruction_inputs_hiddens_outputs(RBM_obj, Total_inputs, plot_choice, Hidden_units_second, hiddens_plotted, numbers_layers, RBM_obj_second)

    print "...finish test second layer"'\n'
