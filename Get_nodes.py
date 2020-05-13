
import numpy as np

class Get_Nodes():

    def __init__(self, batch_size=50, seq_length=5):
        '''
        Initializer function for the Get_Nodes class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        '''
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.nodes = [{} for i in range(batch_size)]

    def reset(self):
        self.nodes = [{} for i in range(self.batch_size)]

    def construct_nodes(self, source_batch):
        '''
        Main function that gets the node info from the batch data
        params:
        source_batch : List of lists of numpy arrays. Each numpy array corresponds to a frame in the sequence.
        '''
        for sequence in range(self.batch_size):
            # source_seq is a list of numpy arrays
            # where each numpy array corresponds to a single frame
            source_seq = source_batch[sequence]
            for framenum in range(self.seq_length):
                # Each frame is a numpy array
                # each row in the array is of the form
                # pedID, x, y
                frame = source_seq[framenum]

                # Add nodes
                for ped in range(frame.shape[0]):
                    pedID = frame[ped, 0]
                    x = frame[ped, 1]
                    y = frame[ped, 2]
                    pos = (x, y)

                    if pedID not in self.nodes[sequence]:
                        node_type = 'H'
                        node_id = pedID
                        node_pos_list = {}
                        node_pos_list[framenum] = pos
                        self.nodes[sequence][pedID] = ST_NODE(node_type, node_id, node_pos_list)

                    else:
                        self.nodes[sequence][pedID].addPosition(pos, framenum)

    def getSequence(self, ind):
        '''
        Gets the data related to the ind-th sequence
        '''
        nodes = self.nodes[ind]

        numNodes = len(nodes.keys())
        list_of_nodes = {}

        retNodes = np.zeros((self.seq_length, numNodes, 2))
        retNodePresent = [[] for c in range(self.seq_length)]

        for i, ped in enumerate(nodes.keys()):
            list_of_nodes[ped] = i
            pos_list = nodes[ped].node_pos_list
            for framenum in range(self.seq_length):
                if framenum in pos_list:
                    retNodePresent[framenum].append(i)
                    retNodes[framenum, i, :] = list(pos_list[framenum])

        return retNodes, retNodePresent

    def getBatch(self):
        return [self.getSequence(ind) for ind in range(self.batch_size)]


class ST_NODE():

    def __init__(self, node_type, node_id, node_pos_list):
        '''
        Initializer function for the ST node class
        params:
        node_type : Type of the node (Human or Obstacle)
        node_id : Pedestrian ID or the obstacle ID
        node_pos_list : Positions of the entity associated with the node in the sequence
        '''
        self.node_type = node_type
        self.node_id = node_id
        self.node_pos_list = node_pos_list

    def getPosition(self, index):
        '''
        Get the position of the node at time-step index in the sequence
        params:
        index : time-step
        '''
        assert(index in self.node_pos_list)
        return self.node_pos_list[index]

    def getType(self):
        '''
        Get node type
        '''
        return self.node_type

    def getID(self):
        '''
        Get node ID
        '''
        return self.node_id

    def addPosition(self, pos, index):
        '''
        Add position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        '''
        assert(index not in self.node_pos_list)
        self.node_pos_list[index] = pos

    def printNode(self):
        '''
        Print function for the node
        For debugging purposes
        '''
        print('Node type:', self.node_type, 'with ID:', self.node_id, 'with positions:', self.node_pos_list.values(), 'at time-steps:', self.node_pos_list.keys())
