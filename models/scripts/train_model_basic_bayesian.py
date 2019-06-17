import torch
from data_generation.generate_data import *
import matplotlib.pyplot as plt
from models.LSTM_VAR.BayesianLSTM import LSTM
from math import floor


#####################
# Set parameters
#####################

# Data params
noise_var = 0
num_datapoints = 300
test_size = 0.2
num_train = int((1 - test_size) * num_datapoints)

# Network params
input_size = 20

per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
# size of hidden layers

h1 = 5
output_dim = 1
num_layers = 1
learning_rate = 1e-3
num_epochs = 650
n_data = 300
dtype = torch.float
length_of_sequences = 10


#####################
# Generate data_handling
#####################

data = np.sin(0.2*np.linspace(0, 200, n_data))
number_of_sequences = n_data - length_of_sequences+ 1

sequences = list()
for sequence_index in range(number_of_sequences):
    sequence = data[sequence_index:sequence_index + length_of_sequences]
    sequences.append(sequence)

sequences = np.array(sequences)



# make training and test sets in torch
training_data = sequences[:, np.newaxis, :]
training_data = np.swapaxes(training_data, axis1=2, axis2=0)
training_data = np.swapaxes(training_data, axis1=2, axis2=1)



training_data_labels = training_data[length_of_sequences-1, : , 0]


plt.plot(training_data[length_of_sequences-1, :, 0])
plt.plot(training_data_labels)
plt.legend()
plt.show()

#####################
# Build model
#####################
batch_size = 10

model = LSTM(1, h1, batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)
loss_fn = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

def data_loader(data, batch_size, random=True):

    number_of_batches = int(floor(data.shape[1] / batch_size))
    if random:
        indexes_of_batch = np.random.choice(range(number_of_batches),
                                            number_of_batches, replace=False)
    else:
        indexes_of_batch = range(number_of_batches)

    for index_batch in indexes_of_batch:
        batch_first_index = index_batch * batch_size
        batch_last_index = (index_batch + 1) * batch_size

        X_numpy = data[:, batch_first_index:batch_last_index, 0]
        X_torch = torch.from_numpy(X_numpy).type(torch.Tensor)

        y_numpy = data[-1, batch_first_index:batch_last_index, 0]
        y_torch = torch.from_numpy(y_numpy).type(torch.Tensor)

        yield X_torch, y_torch

def make_forward_pass(data_loader, model, loss_fn):

    losses = None
    N_data = 0

    for X_train, y_train in data_loader(training_data, batch_size):

        N_data += batch_size
        y_pred = model(X_train)

        if losses is None:
            losses = loss_fn(y_pred, y_train)
        else:
            losses += loss_fn(y_pred, y_train)

    return losses, N_data

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass

    losses, N_data = make_forward_pass(data_loader, model, loss_fn)
    if t % 10 == 0:
        print("Epoch ", t, "MSE: ", losses.item())

    hist[t] = losses.item()

    optimiser.zero_grad()
    losses.backward()
    optimiser.step()

#####################
# Plot preds and performance
#####################


def make_predictions(data_loader, model):
    """
       NOTE: the feature index for the noise is based on the keeped indexes!!
    """
    model.eval()

    y_pred_all = None
    y_test_all = None

    for X_train, y_train in data_loader(training_data, batch_size, random=False):

        y_pred = model(X_train)


        if y_pred_all is None:
            y_pred_all = y_pred.detach().numpy()
            y_test_all = y_train.detach().numpy()

        y_pred_all = np.concatenate([y_pred_all, y_pred.detach().numpy()])
        y_test_all = np.concatenate([y_test_all, y_train.detach().numpy()])

    model.train()

    return y_pred_all, y_test_all




y_pred, y_true = make_predictions(data_loader, model)


plt.plot(y_pred, label="Preds")
plt.plot(y_true, label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()