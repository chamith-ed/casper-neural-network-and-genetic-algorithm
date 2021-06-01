import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from CasperNN import Net, add_neuron
from DataLoader import DataFrameDataset


'''
Step 1: Prepare Data
Returns training, validation and testing sets
'''

def prepare_data():
    # Convert data into dataframe
    eeg_features = pd.read_csv('data/music-eeg-features.csv')
    # drop subject number column
    eeg_features = eeg_features.drop(columns=['subject no.'])
    # print(eeg_features["label"].value_counts())

    # Apply min-max normalisation
    for column in eeg_features.columns[:-1]:
        # the last column is target
        eeg_features[column] = eeg_features.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

    print(eeg_features.describe())

    # split data randomly into training set (80%) and testing set (20%)
    msk = np.random.rand(len(eeg_features)) < 0.8
    train_val_data = eeg_features[msk]
    test_data = eeg_features[~msk]

    # split training set into validation and training set randomly
    # Overall: 60% training, 20% validation, 20% testing
    msk2 = np.random.rand(len(train_val_data)) < 0.75
    train_data = train_val_data[msk2]
    val_data = train_val_data[~msk2]

    # define special train dataset for input into pytorch model
    train_dataset = DataFrameDataset(df=train_data)
    return test_data, train_dataset, val_data


'''
Step 2: Train Casper Network
Returns trained model
'''
def train_casper(train_loader, num_classes, input_size, P, num_epochs, l1, l2, l3):
    # Use GPU if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialise Neural Network
    net = Net(input_size, num_classes)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Initial learning rate of 0.2
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.2)

    all_losses = []

    prev_loss = float('inf')

    # Number of epochs to run before adding new neuron
    time_period = 15 + P
    N = 0
    for epoch in range(num_epochs):
        total = 0
        correct = 0
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            X = batch_x
            Y = batch_y.long()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(X)
            loss = criterion(outputs, Y)
            all_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if epoch == time_period or epoch % 100 == 0:
            # if epoch == time_period:
                _, predicted = torch.max(outputs, 1)
                # calculate and print accuracy
                total = total + predicted.size(0)
                correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
                total_loss = total_loss + loss

        # Check if loss has decreased enough, if not, add neuron
        if epoch == time_period:
            # if prev_loss - prev_loss*0.01 < total_loss and total_loss > 1:
            if prev_loss - prev_loss * 0.01 < total_loss:
                prev_loss = total_loss
                optimizer = add_neuron(net, num_classes, input_size, l1, l2, l3)
                N = net.added_neurons
                print('Neurons added:', N)
            prev_loss = total_loss
            # print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
            #       % (epoch + 1, num_epochs,
            #          total_loss, 100 * correct/total))
            time_period += 15 + P * N

            # if number of epochs to run before adding neuron is bigger than num_epochs, stop training
            if time_period > num_epochs:
                break
        if epoch % 100 == 0:
            print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
                  % (epoch + 1, num_epochs,
                     total_loss, 100 * correct / total))
    # if type(total_loss) is int:
    #     t = total_loss
    # else:
    #     t = total_loss.item()
    #
    # print(t)
    return net


"""
Step 3: Test the neural network

Pass testing/validation data to the built neural network and get its performance
Returns testing/validation loss
"""

def validate_casper(net, val_data, input_size):
    # get testing data
    val_input = val_data.iloc[:, :input_size]
    val_target = val_data.iloc[:, input_size]

    inputs = torch.Tensor(val_input.values).float()
    targets = torch.Tensor(val_target.values - 1).long()
    criterion = nn.CrossEntropyLoss()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    _, predicted = torch.max(outputs, 1)

    total = predicted.size(0)
    correct = predicted.data.numpy() == targets.data.numpy()

    print('Validation Accuracy: %.2f %%' % (100 * sum(correct) / total), loss.item())
    return loss.item()

def test_casper(net, test_data, input_size):
    # get testing data
    test_input = test_data.iloc[:, :input_size]
    test_target = test_data.iloc[:, input_size]

    inputs = torch.Tensor(test_input.values).float()
    targets = torch.Tensor(test_target.values - 1).long()

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    total = predicted.size(0)
    correct = predicted.data.numpy() == targets.data.numpy()

    # print('Testing Accuracy: %.2f %%' % (100 * sum(correct)/total))
    return 100 * sum(correct)/total

'''
Trains the Casper Network and tests it
Prints Testing Accuracy
'''
def main():
    # Data descriptors
    input_size = 25
    num_classes = 3

    # Hyper Parameters
    num_epochs = 500
    batch_size = 15
    P = 5  # Equation used to determine if neuron should be added: 15 + P*Neurons added

    # Prepare Data
    test_data, train_dataset, val_data = prepare_data()

    # define train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train Casper
    net = train_casper(train_loader, num_classes, input_size, P, num_epochs, 0.2, 0.005, 0.001)

    # Validate Casper
    # validate_casper(net, val_data)

    # Test Casper
    acc = test_casper(net, test_data, input_size)
    print('Testing Accuracy: %.2f %%' % acc)


if __name__ == '__main__':
    main()

