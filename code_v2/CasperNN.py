import torch
import torch.nn as nn


# Casper Neural Network
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        # connections from the input and other neurons to a neuron
        self.in_to_neurons = nn.ModuleList()

        # connections from the neurons to the output
        self.neurons_to_out = nn.ModuleList()

        # number of added neurons
        self.added_neurons = len(self.in_to_neurons)

        # connections from input to output
        self.in_to_out = nn.Linear(input_size, num_classes)

        # activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_final = self.in_to_out(x)  # Calculate connection from input to output
        output_list = list()
        output = 0
        for i in range(len(self.in_to_neurons)):
            if i == 0:  # If first added neuron, calculate output straight away
                output = self.sigmoid(self.in_to_neurons[i](x))
                output_list.append(output)

            # If not first neuron, utilise stored outputs
            # of previous neurons and original input to calculate output for the neuron
            if i > 0:
                new_x = x
                # Go through previous neuron outputs and concatenate them
                for j in range(len(output_list)):
                    new_x = torch.cat((new_x, output_list[j]), 1)
                # Use concatenation of previous neurons outputs to calculate output of next neuron
                output = self.sigmoid(self.in_to_neurons[i](new_x))
                # Append output of the neuron to list of outputs to be used for next possible neuron
                output_list.append(output)

            # Add neuron output to total output
            output = self.neurons_to_out[i](output)
            out_final += output

        return out_final


'''
Adds a neuron to the neural network and updates learning rates of connections.
Outputs the optimiser.
'''
def add_neuron(net, num_classes, input_size, l1, l2, l3):
    net.added_neurons += 1
    # New neuron takes input from old neurons and original input
    net.in_to_neurons.append(nn.Linear(input_size + net.added_neurons - 1, 1))
    # New neuron outputs to the number of classes
    net.neurons_to_out.append(nn.Linear(1, num_classes))
    # Set learning rates of weights
    params = [
        # Weight from input to added neuron
        {'params': net.in_to_neurons[net.added_neurons - 1].parameters(), 'lr': l1},  # L1
        # Weights from added neuron to output
        {'params': net.neurons_to_out[net.added_neurons - 1].parameters(), 'lr': l2},  # L2
    ]
    for i in range(net.added_neurons - 1):
        # Weights from input to previously added neurons
        params.append(
            {'params': net.in_to_neurons[i].parameters(), 'lr': l3})  # L3
        # Weights from previously added neurons to output
        params.append(
            {'params': net.neurons_to_out[i].parameters(), 'lr': l3})  # L3

    # Use RMSprop optimiser
    opt = torch.optim.RMSprop(params)

    return opt
