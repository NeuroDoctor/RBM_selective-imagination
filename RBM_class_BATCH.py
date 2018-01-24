import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
import csv
import copy
from numba import vectorize
from numba import jit



class RBM(object):
    """"


    """


    def __init__(self, single_layer, size_hidden, learning_rate = 0.1, alfa = 0.5, sparsety = 0.0, K = 1):

        # other variables
        self.size_hidden = size_hidden
        self.K = K
        self.sparsety = sparsety
        self.learning_rate = learning_rate
        self.alfa = alfa
        self.n = 0
        self.single_layer = single_layer
        self.input_units = 0
        self.reconstructed_matrices = []
        self.reconstructed_all_output = []
        self.single_input_errors = x = [[] for i in range(9)]
        self.single_input_errors_2 = x = [[] for i in range(9)]
        self.FirstSecondErrors = []
        self.FirstErrors = []
        self.SecondErrors = []
        self.epoc_ = 0
        self.epoc_2 = 0

        #INPUT VARIABLES

        self.cumulated_bias_hidden_weights = np.zeros((1,size_hidden))
        self.input = np.zeros((self.input_units))
        self.input_update_weights = np.zeros((self.input_units, size_hidden))
        self.input_update_weights_prev = np.zeros((self.input_units, size_hidden))
        self.cumulated_input_hidden_weights = np.zeros((self.input_units, size_hidden))
        self.input_weights = np.zeros((self.input_units, size_hidden))
        self.rec_input = np.zeros((self.input_units))
        #inputs_biases
        self.cumulated_bias_input_weights = np.zeros((1,size_hidden))
        self.bias_inputs_update_weights_prev = np.zeros((1, size_hidden))
        self.bias_inputs_weights = np.zeros((1, size_hidden))



        #HIDDENS VARIABLES

        self.selected_max_hiddens_activation = []
        self.reconstructed_all_hidden = []

        #hidden_biases
        self.cumulated_bias_hidden_weights = np.zeros((1, self.input_units))
        self.bias_hidden_update_weights_prev = np.zeros((1,self.input_units))
        self.bias_hidden_weights = np.zeros((1, self.input_units))
        self.reconstructed_hidden_bias = [0]






#   FUNCTIONS FOR LOADING IMAGES, GETTING INPUTS, INITIALIZATING WEIGHTS

    def get_images (self, resize_choice, MNIST = 0):

        self.MNIST_value = MNIST

        if MNIST == 1:

            MNIST_matricial_train = np.zeros((60000, 784))
            MNIST_matricial_test = np.zeros((10000, 784))

            with open('mnist_train.csv', 'r') as csv_file:
                n = 0
                for data in csv.reader(csv_file):
                    # The first column is the label
                    label = data[0]

                    # The rest of columns are pixels
                    pixels = data[1:]

                    # Make those columns into a array of 8-bits pixels
                    # This array will be of 1D with length 784
                    # The pixel intensity values are integers from 0 to 255
                    pixels = (np.array(pixels, dtype='float64')) / 255

                    MNIST_matricial_train[n, :] = pixels

                    n += 1

                n = 0

            with open('mnist_test.csv', 'r') as csv_file:
                for data in csv.reader(csv_file):
                    # The first column is the label
                    label = data[0]

                    # The rest of columns are pixels
                    pixels = data[1:]

                    # Make those columns into a array of 8-bits pixels
                    # This array will be of 1D with length 784
                    # The pixel intensity values are integers from 0 to 255
                    pixels = (np.array(pixels, dtype='float64')) / 255

                    MNIST_matricial_test[n, :] = pixels

            originals_matrices = MNIST_matricial_test

            inputs_test_learn = MNIST_matricial_train

            self.originals_matrices = originals_matrices

            self.inputs_test_learn = np.array(inputs_test_learn)

            self.number_input_units = len(inputs_test_learn)

            print "MNIST loaded!" '\n'

        # Important images_foolder#

        # np.load("C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_polygons_inputs.npz")
        # np.load('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\Tests images on RBM\Original_images\Resize 28x28\Original_resize_imgs_28x28.npy')

        elif MNIST == 0:

            originals_matrices = np.load("C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_polygons_inputs.npz")

            inputs_test_learn = np.load("C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_polygons_inputs.npz")

            originals_matrices = [originals_matrices[key] for key in originals_matrices]

            inputs_test_learn = [inputs_test_learn[key] for key in inputs_test_learn]

            for i in range(0, len(inputs_test_learn)):

                inputs_test_learn[i] = np.array(inputs_test_learn[i])



            for i in range(0, len(inputs_test_learn)):

                inputs_test_learn[i] = inputs_test_learn[i].flatten('F')

            self.originals_matrices = originals_matrices

            self.inputs_test_learn = np.array(inputs_test_learn)

            self.number_input_units = len(inputs_test_learn[0])

            print("Polygons images loaded (66 x 66 x 3) " '\n')


            if resize_choice == 1:

                resized_matrices = []

                for i in range(0, len(inputs_test_learn)):

                    img = Image.open("C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\Tests images on RBM\Original_images\Resize 28x28\Old resizing\Original_images_" + str(
                            i + 1) + ".jpg")
                    img.load()
                    data = np.asarray(img, dtype="float64")
                    data = data / 255
                    # data= np.transpose(data, (1, 0, 2))
                    resized_matrices.append(data)


                # np.random.shuffle(inputs_test_learn)

                inputs_test_learn = resized_matrices

                for i in range(0, len(inputs_test_learn)):
                    inputs_test_learn[i] = np.array(inputs_test_learn[i])


                for i in range(0, len(inputs_test_learn)):

                    inputs_test_learn[i] = inputs_test_learn[i].flatten('F')

                print("resize of polygons = from 66 x 66 to 28 x 28 " '\n')

                self.originals_matrices = originals_matrices

                self.inputs_test_learn = np.array(inputs_test_learn)

                self.number_input_units = len(inputs_test_learn[0])

    def variables_changes_for_images(self, single_layer, hidden_output_first = 0):


        #dynamical initialization of RBM class depending on images..


        if single_layer == 1:

            self.input_units = self.inputs_test_learn[0].shape[0]

        elif single_layer == 2:

            self.input_units = hidden_output_first.shape[1]



        self.input = np.zeros((self.input_units))
        self.input_update_weights = np.zeros((self.input_units, self.size_hidden))
        self.input_update_weights_prev = np.zeros((self.input_units, self.size_hidden))
        self.cumulated_input_hidden_weights = np.zeros((self.input_units, self.size_hidden))
        self.input_weights = np.zeros((self.input_units, self.size_hidden))
        self.rec_input = np.zeros((self.input_units))

        #hidden_biases
        self.cumulated_bias_hidden_weights = np.zeros((1, self.input_units))
        self.bias_hidden_update_weights_prev = np.zeros((1,self.input_units))
        self.bias_hidden_weights = np.zeros((1, self.input_units))


    def get_input(self,single_layer, MNIST, batch_single, Batch_size_first,hiddens_second_input = 0, test = 0):

        self.MNIST_value = MNIST

        self.epoc_ += 1

        if single_layer == 1:

            self.input = self.inputs_test_learn

        elif single_layer == 2:

            self.input = hiddens_second_input

        if MNIST == 1 and test == 0:

            inf_lim_input = batch_single*Batch_size_first
            sup_lim_input = ((batch_single*Batch_size_first) + Batch_size_first)

            self.input = self.inputs_test_learn[inf_lim_input:sup_lim_input,:]


        if self.epoc_ == 1:

            self.input_first = self.input

    def initialization_weights(self, size_input, size_hidden):

        self.input_weights = np.random.uniform(-0.1, 0.1, (size_input, size_hidden))

        self.bias_inputs_weights = np.random.uniform(-0.1, 0.1,(1, size_hidden))

        self.bias_hidden_weights = np.random.uniform(-0.1, 0.1,(1, size_input))


# Save and Load weights of first layer to do not start learning every time

    def save_weights(self, save_choice):

        if save_choice == 1:

            np.save('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_first_layer_weights', self.input_weights)
            np.save('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_first_layer_Bias_inputs_weights', self.bias_inputs_weights)
            np.save('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_first_layer_Bias_hidden_weights', self.bias_hidden_weights)
            np.save('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\hidden_output_first_layer', self.hidden_output)

    def load_weights(self, load_choice):

        if load_choice == 1:

            self.input_weights = np.load('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_first_layer_weights.npy')
            self.bias_inputs_weights = np.load('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_first_layer_Bias_inputs_weights.npy')
            self.bias_hidden_weights = np.load('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\RBM_first_layer_Bias_hidden_weights.npy')
            self.hidden_output = np.load('C:\Users\streg\Google Drive\CNR\Reti Neurali\Python\Pycharm\hidden_output_first_layer.npy')





#    SPREAD, RECONSTRUCTION FUNCTIONS



    def get_output_first_layer(self, test = 0, single_layer = 0):


        self.hidden_pot = np.dot(self.input, self.input_weights)

        self.hidden_pot = self.hidden_pot + self.bias_inputs_weights

        self.hidden_output = 1 / (1 + np.exp(-(self.hidden_pot)))


        # SMALL BARBA-TRICK TO HELP THE nn TO ACTIVE SPECIFIC NEURONS (LOWER THE TRESHOLD)

        # if single_layer == 2 and test == 1:
        #
        #     random_treshold = 0.001
        #
        #     self.hidden_output[self.hidden_output > random_treshold] = 1
        #
        #     self.hidden_output[self.hidden_output < random_treshold] = 0



        if self.epoc_ == 1 and test == 0:

            self.hidde_output_first = self.hidden_output

            random_treshold = np.random.random_sample()  # noise, case

            # SMALL BARBA-TRICK TO HELP THE nn TO ACTIVE SPECIFIC NEURONS (LOWER THE TRESHOLD)

            # if single_layer == 2:
            #
            #     random_treshold = 0.5


            self.hidden_output[self.hidden_output > random_treshold] = 1

            self.hidden_output[self.hidden_output < random_treshold] = 0

            #IMPLEMENTATION OF SPARSETY

            current_sparsety = (np.sum(((np.sum((self.hidden_output), axis = 0)) / self.input.shape[0]))) / self.hidden_output.shape[1]

            current_target = (current_sparsety - self.sparsety)

            self.penalty = self.learning_rate * current_target * self.K

            if current_sparsety <= self.sparsety:

                self.penalty = 0 * self.penalty # if current sparsety is lower then target sparsety: penalty = 0

            self.hidde_output_first = self.hidden_output


        if single_layer == 2: #

            self.real_activation_second_hidden = copy.deepcopy(self.hidden_output) # to save variable for reconstruction of second RF


        self.epoc_ += 1

    def perceptron_selective_reconstruction(self,proprierty):

        weights_first_layer = np.array([-1, -1])
        weights_inibitor_layer = np.array([[0, 0, 0, -1, -1, -1], [-1, -1, -1, 0, 0, 0]])
        inibitor_layer = np.array([1, 1])

        if proprierty == 0:
            input_ = np.array([1, 1])

        elif proprierty == 1:
            input_ = np.array([1, 0])
        elif proprierty == 2:
            input_ = np.array([0, 1])
        elif proprierty > 2:
            input_ = np.array([0, 0])

        inibitor_layer += input_ * weights_first_layer

        self.selection_bias = np.dot(inibitor_layer, weights_inibitor_layer)


        for i in range(0, self.hidden_output.shape[0]):

            self.hidden_output[i,:] += self.selection_bias
            self.real_activation_second_hidden[i,:] += self.selection_bias

            self.hidden_output[self.hidden_output < 0] = 0
            self.real_activation_second_hidden[self.real_activation_second_hidden < 0] = 0

    def selective_learning(self, learning, proprierty):

        if learning == 1: # to chose which neurons process which attribute (green, red, blu, square, circle, bar)

            self.hidde_output_first[:,:] = 0

            self.hidde_output_first[0, [0, 3]] = 1
            self.hidde_output_first[1, [1, 3]] = 1
            self.hidde_output_first[2, [2, 3]] = 1
            self.hidde_output_first[3, [0, 4]] = 1
            self.hidde_output_first[4, [1, 4]] = 1
            self.hidde_output_first[5, [2, 4]] = 1
            self.hidde_output_first[6, [0, 5]] = 1
            self.hidde_output_first[7, [1, 5]] = 1
            self.hidde_output_first[8, [2, 5]] = 1

        # elif learning == 0 and proprierty == 0: # select the shape for recoonstruction
        #
        #     #SMALL TRICK TO SHOW SINGLE RECEPTIVE FIELDS
        #
        #     # self.hidden_output[:, :] = 0
        #     #
        #     # for i in range(0,6):
        #     #
        #     #     self.hidden_output[i, i] = 1
        #
        #     #   SPREAD RESULT MANIPULATION (CHANGING SECOND_HIDDEN ACTIVATION)
        #     self.hidden_output[0, [0, 1, 2]] = 0
        #     self.hidden_output[1, [0, 1, 2]] = 0
        #     self.hidden_output[2, [0, 1, 2]] = 0
        #     self.hidden_output[3, [0, 1, 2]] = 0
        #     self.hidden_output[4, [0, 1, 2]] = 0
        #     self.hidden_output[5, [0, 1, 2]] = 0
        #     self.hidden_output[6, [0, 1, 2]] = 0
        #     self.hidden_output[7, [0, 1, 2]] = 0
        #     self.hidden_output[8, [0, 1, 2]] = 0
        #     # SAME THING CONSIDERING THE GRAPHICAL VARIABLE O SPREAD RESULT
        #     self.real_activation_second_hidden[0, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[1, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[2, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[3, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[4, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[5, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[6, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[7, [0, 1, 2]] = 0
        #     self.real_activation_second_hidden[8, [0, 1, 2]] = 0
        #
        # elif learning == 0 and proprierty == 1: # select the color for reconstruction
        #
        #     #   SPREAD RESULT MANIPULATION (CHANGING SECOND_HIDDEN ACTIVATION)
        #     self.hidden_output[0, [3, 4, 5]] = 0
        #     self.hidden_output[1, [3, 4, 5]] = 0
        #     self.hidden_output[2, [3, 4, 5]] = 0
        #     self.hidden_output[3, [3, 4, 5]] = 0
        #     self.hidden_output[4, [3, 4, 5]] = 0
        #     self.hidden_output[5, [3, 4, 5]] = 0
        #     self.hidden_output[6, [3, 4, 5]] = 0
        #     self.hidden_output[7, [3, 4, 5]] = 0
        #     self.hidden_output[8, [3, 4, 5]] = 0
        #     # SAME THING CONSIDERING THE GRAPHICAL VARIABLE O SPREAD RESULT
        #     self.real_activation_second_hidden[0, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[1, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[2, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[3, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[4, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[5, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[6, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[7, [3, 4, 5]] = 0
        #     self.real_activation_second_hidden[8, [3, 4, 5]] = 0



    def input_reconstruction(self, test=0, print_error = 0):

        rec_pot = np.dot(self.hidden_output, self.input_weights.T)

        rec_pot = rec_pot + self.bias_hidden_weights

        self.rec_input = 1 / (1 + np.exp(-(rec_pot)))

        self.input = self.rec_input

        #EPOC ERROR CALCULATION

        self.error_1 = np.abs(self.input_first - self.rec_input)

        self.errors_for_input = self.error_1.sum(axis=1) / self.error_1.shape[1]

        error_elements = (self.error_1.shape[0]) * (self.error_1.shape[1])

        self.avg_errors_1 = self.error_1.sum() / error_elements

        self.percent_error_1 = self.avg_errors_1 * 100


        if print_error == 1:

            for inp in range(0,self.error_1.shape[0]):

                print "input n ", inp, " reconstruction error = ", np.around((self.errors_for_input[inp] * 100),decimals=5), '\n'


#     UPDATE OF WEIGHTS, LEARNING


    def update_weights_input_hidden(self):

        PositiveProduct = np.dot(self.input_first.T, self.hidde_output_first)

        NegativeProduct = np.dot(self.input.T, self.hidden_output)

        self.input_update_weights = self.learning_rate * ((PositiveProduct - NegativeProduct) / self.input.shape[0]) + self.alfa * self.input_update_weights_prev


        self.bias_inputs_update_weights = self.learning_rate / self.input.shape[0] * (sum(self.hidde_output_first) - sum(self.hidden_output)) + self.alfa * self.bias_inputs_update_weights_prev

        self.bias_inputs_update_weights =  self.bias_inputs_update_weights - self.penalty.reshape((-1, 1))



        self.bias_hidden_update_weights = self.learning_rate / self.input.shape[0] * (sum(self.input_first) - sum(self.input)) + self.alfa * self.bias_hidden_update_weights_prev


        self.cumulated_input_hidden_weights += self.input_update_weights

        self.cumulated_bias_input_weights += self.bias_inputs_update_weights

        self.cumulated_bias_hidden_weights += self.bias_hidden_update_weights


        self.input_update_weights_prev = self.input_update_weights

        self.bias_inputs_update_weights_prev = self.bias_inputs_update_weights

        self.bias_hidden_update_weights_prev = self.bias_hidden_update_weights

        self.epoc_ = 0


    def update_BATCH_input_hidden(self, Batch_size):




        self.input_weights += self.cumulated_input_hidden_weights #/ Batch_size

        self.bias_inputs_weights += self.cumulated_bias_input_weights #/ Batch_size

        self.bias_hidden_weights += self.cumulated_bias_hidden_weights #/ Batch_size

        self.cumulated_input_hidden_weights = 0

        self.cumulated_bias_input_weights = 0

        self.cumulated_bias_hidden_weights = 0





# FUNCTION THAT RETURNS GRAPHS AND RECEPTIVE FIELDS
    def Graphical_reconstruction_inputs_hiddens_outputs(self,RBM_obj, Total_inputs, plot_choice, Hidden_units_second, hiddens_plotted, numbers_layers, RBM_obj_second =0):


        for inp in range(0,Total_inputs):

            #print "input n ", inp, " reconstruction error = ", np.around((self.errors_for_input[inp] * 100), decimals = 5), '\n'

            RBM_obj.Polygons_recostructions(inp)
            RBM_obj.Polygons_hidden_recostruction(inp, plot_choice, hiddens_plotted)

            if numbers_layers == 2:
                RBM_obj.Polygons_hidden_second_recostruction(RBM_obj, RBM_obj_second, Hidden_units_second)


            RBM_obj.Graphical_poly_rec(inp, plot_choice, hiddens_plotted, numbers_layers)
            RBM_obj.Graphical_hidden_rec(1, inp, numbers_layers)

            if numbers_layers == 2:
                RBM_obj.Graphical_hidden_rec(2, inp, numbers_layers, RBM_obj_second)

            #RBM_obj.reconstructed_all_hidden = []


    # MATRICIAL RECONSTRUCTIONS (INPUTS, HIDDENS, OUTPUTS)

    def Polygons_recostructions(self, single_input):

        if self.MNIST_value == 0:
            self.image_side = np.sqrt(((self.rec_input.shape[1]) / 3))

        elif self.MNIST_value == 1:
            self.image_side = np.sqrt(((self.rec_input.shape[1]) / 1))


        reconstructed_single_input = self.rec_input
        reconstructed_single_original = self.input_first


        self.n += 1



        if self.MNIST_value == 0:
            self.original = reconstructed_single_original[single_input].reshape([int(self.image_side), int(self.image_side), 3], order='F')

        elif self.MNIST_value == 1:
            self.original = reconstructed_single_original[single_input].reshape([int(self.image_side), int(self.image_side)])


        # change of values ina 0-1 range
        max = np.max(self.original)
        min = np.min(self.original)
        m = interp1d([min, max], [0, 1])
        #self.original = m(original)

        if self.MNIST_value == 0:
            self.rec_matrix = reconstructed_single_input[single_input].reshape([int(self.image_side), int(self.image_side), 3], order='F')

        elif self.MNIST_value == 1:
            self.rec_matrix = reconstructed_single_input[single_input].reshape([int(self.image_side), int(self.image_side)])


        #change of values ina 0-1 range
        max = np.max(self.rec_matrix)
        min = np.min(self.rec_matrix)
        m = interp1d([min, max], [0, 1])
        # self.rec_matrix = m(self.rec_matrix)

        self.reconstructed_matrices.append(self.rec_matrix)

    def Polygons_hidden_recostruction(self, single_inp, personal_plotting_choice, hiddens_plotted):

        self.hiddens_activation_for_inp = self.hidden_output[single_inp,:]

        if self.single_layer == 2:

            self.hiddens_activation_for_inp == self.input


        #loop for reconstruction of each Hidden receptors field...

        for i in range(0, len(self.hiddens_activation_for_inp)):

            if self.hiddens_activation_for_inp[i] > 0.1:

                self.selected_max_hiddens_activation.append(i)

        if personal_plotting_choice == 1:

            self.selected_max_hiddens_activation = self.hiddens_activation_for_inp.argsort()[::-1][:hiddens_plotted]


        for i in range(0, len(self.selected_max_hiddens_activation)):

                self.rec_single_hidden = self.input_weights[:, self.selected_max_hiddens_activation[i]] #take single hidden receptor field

                # change of values ina 0-1 range

                max = np.max(self.rec_single_hidden)
                min = np.min(self.rec_single_hidden)
                m = interp1d([min, max],[0,1])
                self.rec_single_hidden = m(self.rec_single_hidden)


                #reconstruction of hiddens matrices

                if self.MNIST_value == 0:
                    self.recostructed_single_hidden = self.rec_single_hidden.reshape([int(self.image_side), int(self.image_side), 3], order='F')

                elif self.MNIST_value == 1:
                    self.recostructed_single_hidden = self.rec_single_hidden.reshape([int(self.image_side), int(self.image_side)])

                self.reconstructed_all_hidden.append(self.recostructed_single_hidden)



        # bias receptor field reconstruction
        if self.MNIST_value == 0:

            self.reconstructed_hidden_bias[0] = (self.bias_hidden_weights.reshape([int(self.image_side), int(self.image_side), 3], order='F'))

        elif self.MNIST_value == 1:

            self.reconstructed_hidden_bias[0] = (self.bias_hidden_weights.reshape([int(self.image_side), int(self.image_side)]))

        max = np.max(self.reconstructed_hidden_bias)
        min = np.min(self.reconstructed_hidden_bias)
        m = interp1d([min, max], [0, 1])
        self.reconstructed_hidden_bias = m(self.reconstructed_hidden_bias)

    def Polygons_hidden_second_recostruction(self, RBM_obj_first, RBM_obj_second, Hidden_units_second):

        output_second_RBM_initital = copy.deepcopy(RBM_obj_second.real_activation_second_hidden)

        # simulated reconstruction of input activing specific second_hiddens_neurons


        output_second_RBM = copy.deepcopy(RBM_obj_second.hidden_output)

        output_second_RBM[:, :] = 0

        for i in range(0, Hidden_units_second):

            output_second_RBM[i, i] = 1

        rec_hidden_first_RBM = np.dot(output_second_RBM, RBM_obj_second.input_weights.T)

        rec_hidden_first_RBM_potential = rec_hidden_first_RBM + RBM_obj_second.bias_hidden_weights

        rec_hidden_first_RBM = 1 / (1 + np.exp(-(rec_hidden_first_RBM_potential)))

        rec_input_first_RBM = np.dot(rec_hidden_first_RBM, RBM_obj_first.input_weights.T)

        rec_input_first_RBM_potential = rec_input_first_RBM + RBM_obj_first.bias_hidden_weights

        rec_input_first_RBM = 1 / (1 + np.exp(-(rec_input_first_RBM_potential)))

        self.receptive_fields_Second_hidden = []

        # to extract all of second_RBM_hiddens receptive fields

        for i in range(0, Hidden_units_second + 1):
            single_hidden_Second_receptive_field = rec_input_first_RBM[i, :]

            single_matrix_hidden_second = single_hidden_Second_receptive_field.reshape([int(RBM_obj_first.image_side), int(RBM_obj_first.image_side), 3], order='F')

            self.receptive_fields_Second_hidden.append(single_matrix_hidden_second)

        # to extract second_RBM_hiddens those partecipate to specific input_reconstruction

        self.selected_second_hiddens = [[] for _ in range(9)]

        for h in range(0, len(self.selected_second_hiddens)):

            single_input_hidden_rec_fields = output_second_RBM_initital[h, :]

            for n in range(0, len(single_input_hidden_rec_fields)):

                if single_input_hidden_rec_fields[n] > 0.1:
                    self.selected_second_hiddens[h].append(n)

        self.receptive_fields_for_inp = [[] for _ in range(9)]

        for i in range(0, len(self.selected_second_hiddens)):

            for l in range(0, len(self.selected_second_hiddens[i])):

                coupled_indices = self.selected_second_hiddens[i]

                coupled_index = coupled_indices [l]

                self.receptive_fields_for_inp[i].append(self.receptive_fields_Second_hidden[coupled_index])


        for i in range(0,len(self.receptive_fields_for_inp)):

            self.receptive_fields_for_inp[i].append(self.receptive_fields_Second_hidden[-1])

    # GRAPHICAL RECONSTRUCTIONS (INPUTS, HIDDENS, OUTPUTS)

    def Graphical_poly_rec(self, single_input, plot_choice, hiddens_plotted,numbers_layers):

        #VERTICAL LIMITS
        self.vertical_limit_graphic = 7

        if numbers_layers == 2:

            self.vertical_limit_graphic = 11


        #ORIZONTAL LIMITS
        self.oriz_limit_graphic = hiddens_plotted

        if hiddens_plotted < 7:
            self.oriz_limit_graphic = 7


        #PLOT FIRST IMAGE (ORIGINAL)

        self.figures = plt.subplot2grid((self.vertical_limit_graphic, self.oriz_limit_graphic), (0, 0), colspan = 3, rowspan = 3)

        plt.imshow(self.original)

        self.figures.set_title(' Original image ')

        plt.axis('off')

        # PLOT SECOND IMAGE (RECONSTRUCTION)

        self.figures = plt.subplot2grid((self.vertical_limit_graphic, self.oriz_limit_graphic), (0, self.oriz_limit_graphic - 3), colspan = 3, rowspan = 3)

        plt.imshow(self.rec_matrix)

        self.figures.set_title('Rec image with Error = ' + str((np.around(self.errors_for_input[single_input], decimals=3, out=None) * 100)) + str("%"))

        plt.axis('off')


    def Graphical_hidden_rec(self,single_layer, single_inp,numbers_layers,RBM_obj_second = 0):

        shift = 0
        unit_correction = 0

        if single_layer == 2:

            shift = 4
            unit_correction = 1


        #Activation plot of hiddens

        self.figures = plt.subplot2grid((self.vertical_limit_graphic, self.oriz_limit_graphic), (4 + shift, 0), colspan = self.oriz_limit_graphic)

        Data_to_plot_Y = self.hidden_output[single_inp, :]

        if single_layer == 2:

            Data_to_plot_Y = RBM_obj_second.real_activation_second_hidden[single_inp, :]

        indices_hiddens_data_to_plot = []

        for i in range(len(Data_to_plot_Y)):

            if Data_to_plot_Y[i] > 0.1:

                indices_hiddens_data_to_plot.append(i)


        plt.bar(range(len(Data_to_plot_Y)), Data_to_plot_Y)
        plt.xlim([0, len(Data_to_plot_Y)])
        plt.ylim(0, 1)
        plt.tick_params(axis='x',labelbottom='off')

        if single_layer == 1:

            self.figures = plt.title(" Hiddens activated ")

        if single_layer == 2:

            self.figures = plt.title(" Hiddens second layer activated ")


        if single_layer == 2:

            self.reconstructed_all_hidden = copy.deepcopy(self.receptive_fields_for_inp[single_inp])


        #receptor fields of hiddens - plot

        pos_fig = 0

        for n in range(0, (len(self.reconstructed_all_hidden) - 1)):

            single_hidden_unit = self.reconstructed_all_hidden[n]

            pos_fig = n

            self.figures = plt.subplot2grid((self.vertical_limit_graphic, self.oriz_limit_graphic), (6 + shift, pos_fig))

            if n != (len(self.reconstructed_all_hidden) - 1):

                if single_layer == 1:

                    self.figures = plt.title("H. " + str(self.selected_max_hiddens_activation[n]))

                elif single_layer == 2:

                    self.figures = plt.title("H. " + str(indices_hiddens_data_to_plot[n]))


            plt.imshow(single_hidden_unit)

            plt.axis('off')

        # plot of bias

        self.figures = plt.subplot2grid((self.vertical_limit_graphic, self.oriz_limit_graphic), (6 + shift, pos_fig + 1))

        self.figures = plt.title("b. ")

        if single_layer == 1:

            plt.imshow(self.reconstructed_hidden_bias[0])

            plt.axis('off')

        if single_layer == 2:

            plt.imshow(self.reconstructed_all_hidden[-1])



        self.selected_max_hiddens_activation = []

        self.reconstructed_all_hidden = []

        if numbers_layers == 1:

            plt.show()



        elif numbers_layers == 2:

            if single_layer == 2:

                plt.show()


    def Plot_errors(self, number_layers):


        FirstLayerErrors = self.FirstErrors

        epoc_first = len(FirstLayerErrors)

        SecondLayerErrors = self.SecondErrors

        epoc_second = len(SecondLayerErrors)


        # Inputs reconstruction plot

        plt.figure(2)

        plt.subplot(111)

        # if number_layers == 1:
        #     plt.subplot(111)
        # elif number_layers == 2:
        #     plt.subplot(121)


        plt.plot(range(0, epoc_first),FirstLayerErrors, 'r', label=' AVG error (Inputs)')

        # TO PLOT ALL LINES OF INPUTS RECONSTRUCTIONS
        # for i in range(0, len(self.single_input_errors)):
        #
        #     plt.plot(range(0, epoc_first), self.single_input_errors[i], 'b', label = (' Input n. ' + str(i) + ' error'))


        plt.xlabel('Epoc')
        plt.ylabel('Reconstruction error')
        plt.title('Inputs reconstruction error')
        plt.text((epoc_first * 0.6), (np.max(FirstLayerErrors) * 0.3), ' Final rec. error = ' + str(np.around((FirstLayerErrors[-1]), decimals = 5)), fontsize= 12)
        plt.legend(loc='upper right')
        # Hiddens reconstruction plot

        # if number_layers == 2:

            # plt.subplot(122)
            #
            # plt.figure(2)
            #
            # plt.plot(range(0, epoc_second),SecondLayerErrors, 'r', label=' AVG error (Hiddens) ')
            #
            # # TO PLOT ALL LINES OF HIDDENS RECONSTRUCTIONS
            # # for i in range(0, len(self.single_input_errors_2)):
            # #
            # #     plt.plot(range(0, epoc_second), self.single_input_errors_2[i], 'b', label=(' Hidden n. ' + str(i) + ' error'))
            #
            # plt.xlabel('Epoc')
            # plt.ylabel('Reconstruction error')
            # plt.title('Hiddens reconstruction error')
            # plt.text((epoc_second * 0.6), (np.max(SecondLayerErrors) * 0.7), ' Final rec. error = ' + str(np.around((SecondLayerErrors[-1]), decimals = 5)), fontsize=12)
            # plt.legend(loc='upper right')

        plt.show()






