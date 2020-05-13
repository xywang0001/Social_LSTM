'''
Baseline implementation: https://github.com/ajvirgona/social-lstm-pytorch/tree/master/social_lstm
By Author: Anirudh Vemula
Date: 13th June 2017

Improvements: 1\ Adjust test.py and train.py to accommodate vanilla lstm-> vlstm_train.py, vlstm_test.py
              2\ Adjust optimizer
              3\ reimplementing model.py and vlstm_model.py
              4\ Add Animation.py
'''

import torch
import torch.nn as nn
from torch.autograd import Variable


class VLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(VLSTM, self).__init__()

        self.args = args
        self.infer = infer

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, nodes, nodesPresent, hidden_states, cell_states):
        '''
        Forward pass for the model
        params:
        nodes: Input positions
        nodesPresent: Peds present in each frame
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # Number of peds in the sequence
        numNodes = nodes.size()[1]

        # Construct the output variable
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size)).cuda()

        # For each frame in the sequence
        for framenum in range(self.seq_length):
            # Peds present in the current frame
            nodeIDs = nodesPresent[framenum]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue

            # List of nodes
            list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda())

            # Select the corresponding input positions
            nodes_current = torch.index_select(nodes[framenum], 0, list_of_nodes)

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, list_of_nodes)
            cell_states_current = torch.index_select(cell_states, 0, list_of_nodes)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))

            # One-step of the LSTM
            h_nodes, c_nodes = self.cell(input_embedded, (hidden_states_current, cell_states_current))

            # Compute the output
            outputs[framenum*numNodes + list_of_nodes.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[list_of_nodes.data] = h_nodes
            cell_states[list_of_nodes.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size).cuda())
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
