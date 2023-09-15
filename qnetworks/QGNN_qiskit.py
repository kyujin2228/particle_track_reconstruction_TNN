import tensorflow as tf
# import tensorflow_quantum as tfq
import qiskit
import numpy as np
import cirq
from qcircuits.QCircuit import QCircuit
import qiskit
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    BCELoss,
    Flatten,
    Sigmoid,
    Sequential,
    ReLU,
)
import torch
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile

###############################################################################
class Rescale01(Module):
    def __init__(self):
        super(Rescale01, self).__init__()

    def forward(self, X):
        X = torch.div(
                torch.subtract(
                    X, 
                    torch.min(X)
                ), 
                torch.subtract(
                    torch.max(X), 
                    torch.min(X)
                )
                # ),
            # lambda: X
        )
        return X
###############################################################################
class EdgeNet(Module):
    def __init__(self):
        super(EdgeNet, self).__init__()

        self.n_layers = GNN.config['EN_qc']['n_layers']
        self.n_qubits = GNN.config['EN_qc']['n_qubits']
        self.input_size=(GNN.config['hid_dim']+3)*2

        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(IEC_id=GNN.config['EN_qc']['IEC_id'],
            PQC_id=GNN.config['EN_qc']['PQC_id'],
            MC_id=GNN.config['EN_qc']['MC_id'],
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01)
        
        self.model_circuit, self.IEC_circuit_parameters, self.PQC_circuit_parameters=qc.model_circuit()
        self.obs1,self.obs2,self.obs3= qc.measurement_operators()

        if 'dp_noise' in GNN.config['EN_qc'].keys():
            dp_noise = GNN.config['EN_qc']['dp_noise']
        else:
            # set noise to None if not specified 
            dp_noise = None

        # Prepare PQC layer
        if (dp_noise!=None):
            noise_model = NoiseModel()
            error1 = depolarizing_error(dp_noise, 1)
            # error2 = depolarizing_error(dp_noise, 2)
            noise_model.add_basis_gates('ry')
            noise_model.add_all_qubit_quantum_error(error1, [ 'id', 'ry'])
            # noise_model.add_all_qubit_quantum_error(error2, [ 'cx'])
            sim_noise=AerSimulator(noise_model=noise_model)
            self.model_circuit = transpile(self.model_circuit, sim_noise)
            print('noise simulation')
        else: 
            raise ValueError('Wrong PQC Specifications!')

        self.qnn = EstimatorQNN(
            circuit=self.model_circuit,
            observables=[self.obs1,self.obs2,self.obs3],
            input_params=self.IEC_circuit_parameters,
            weight_params=self.PQC_circuit_parameters,
            input_gradients=True,
        )    

        self.input_layer = Sequential(
            Linear(self.input_size,self.n_qubits),
            ReLU(),
            Rescale01(),
        )

        # Classical readout layer
        self.readout_layer = Sequential(
            Linear(3,1),
            Sigmoid(),
        )

    def forward(self,X, Ri, Ro):
        '''forward pass of the edge network. '''
        # Constrcu the B matrix
        bo = torch.matmul(torch.transpose(Ro,0,1),X)
        bi = torch.matmul(torch.transpose(Ri,0,1),X)
        # Shape of B = N_edges x 6 (2x (3 + Hidden Dimension Size))
        # each row consists of two node that are connected in the input graph.
        B  = torch.cat((bo, bi), 1) # n_edges x 6, 3-> r,phi,z 

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        input_to_circuit = self.input_layer(B) * np.pi

        # Get expectation values for all edges
        if GNN.config['EN_qc']['repetitions']==0:
            # exps = self.exp_layer(
            #     self.model_circuit,
            #     operators=self.measurement_operators,
            #     symbol_names=self.symbol_names,
            #     symbol_values=circuit_data
            # )
            exps=TorchConnector(self.qnn)(input_to_circuit)
            # exps=self.exp_layer(input_to_circuit)
        else:
            exps = self.exp_layer(
                self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=GNN.config['EN_qc']['repetitions']
            )
    
        # Return the output of the final layer
        # print(exps.shape)
        return self.readout_layer(exps)

class NodeNet(Module):
    def __init__(self):
        super(NodeNet, self).__init__()
        
        self.n_layers = GNN.config['NN_qc']['n_layers']
        self.n_qubits = GNN.config['NN_qc']['n_qubits']
        self.input_size=(GNN.config['hid_dim']+3)*3

        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(
            IEC_id=GNN.config['NN_qc']['IEC_id'],
            PQC_id=GNN.config['NN_qc']['PQC_id'],
            MC_id=GNN.config['NN_qc']['MC_id'],
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01
        )
        self.model_circuit, self.IEC_circuit_parameters, self.PQC_circuit_parameters=qc.model_circuit()
        self.obs1,self.obs2,self.obs3= qc.measurement_operators()

        if 'dp_noise' in GNN.config['EN_qc'].keys():
            dp_noise = GNN.config['EN_qc']['dp_noise']
        else:
            # set noise to None if not specified 
            dp_noise = None
        
        # Prepare PQC layer
        if (dp_noise!=None):
            noise_model = NoiseModel()
            error1 = depolarizing_error(dp_noise, 1)
            # error2 = depolarizing_error(dp_noise, 2)
            noise_model.add_basis_gates('ry')
            noise_model.add_all_qubit_quantum_error(error1, [ 'id', 'ry'])
            # noise_model.add_all_qubit_quantum_error(error2, [ 'cx'])
            sim_noise=AerSimulator(noise_model=noise_model)
            self.model_circuit = transpile(self.model_circuit, sim_noise)

        else: 
            raise ValueError('Wrong PQC Specifications!')

        self.qnn = EstimatorQNN(
            circuit=self.model_circuit,
            observables=[self.obs1,self.obs2,self.obs3],
            input_params=self.IEC_circuit_parameters,
            weight_params=self.PQC_circuit_parameters,
            input_gradients=True,
        )   


        self.input_layer = Sequential(
            Linear(self.input_size,self.n_qubits),
            ReLU(),
            Rescale01(),
        )
        self.readout_layer = Sequential(
            Linear(3,GNN.config['hid_dim']),
            ReLU(),
            Rescale01(),
        )


    def forward(self, X, e, Ri, Ro):
        '''forward pass of the node network. '''

        # The following lines constructs the M matrix
        # M matrix contains weighted averages of input and output nodes
        # the weights are the edge probablities.
        bo  = torch.matmul(torch.transpose(Ro,0,1), X)
        bi  = torch.matmul(torch.transpose(Ri,0,1), X) 
        Rwo = Ro * e[:,0]
        Rwi = Ri * e[:,0]
        mi = torch.matmul(Rwi, bo)
        mo = torch.matmul(Rwo, bi)
        # Shape of M = N_nodes x (3x (3 + Hidden Dimension Size))
        # mi: weighted average of input nodes
        # mo: weighted average of output nodes
        M = torch.cat((mi, mo, X),1)

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        input_to_circuit = self.input_layer(M) * np.pi

        # # Combine input data with parameters in a single circuit_data matrix
        # circuit_data = tf.concat(
        #     [
        #         input_to_circuit, 
        #         tf.repeat(self.params,repeats=input_to_circuit.shape[0],axis=0)
        #     ],
        #     axis=1
        # )        

        # Get expectation values for all nodes
        if GNN.config['NN_qc']['repetitions']==0:
            # exps = self.exp_layer(self.model_circuit,
            #     operators=self.measurement_operators,
            #     symbol_names=self.symbol_names,
            #     symbol_values=circuit_data)
            exps=TorchConnector(self.qnn)(input_to_circuit)

        else:
            exps = self.exp_layer(self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=GNN.config['NN_qc']['repetitions'])

        # Return the output of the final layer
        return self.readout_layer(exps)

###############################################################################
class GNN(Module):
    def __init__(self):
        ''' Init function of GNN, inits all GNN blocks. '''
        # super(GNN, self).__init__(name='GNN')
        super(GNN, self).__init__()
        # Define Initial Input Layer
        self.input_size=3
        self.output_size=GNN.config['hid_dim']
        self.InputNet =  Sequential(
            Linear(self.input_size,self.output_size),
            ReLU(),
        )
        # self.InputNet =  tf.keras.layers.Dense(
        #     GNN.config['hid_dim'], input_shape=(3,),
        #     activation='relu',name='InputNet'
        # )
        self.EdgeNet  = EdgeNet()
        self.NodeNet  = NodeNet()
        self.n_iters  = GNN.config['n_iters']
    
    # def call(self, graph_array):
    def forward(self, graph_array):
        ''' forward pass of the GNN '''
        # decompose the graph array
        # data를 torch로 바꿔야함
        X, Ri, Ro = graph_array
        # print(type(X))
        X=torch.tensor(X)
        Ri=torch.tensor(Ri)
        Ro=torch.tensor(Ro)

        # execute InputNet to produce hidden dimensions
        H = self.InputNet(X)
        # add new dimensions to original X matrix
        # H = tf.concat([H,X],axis=1)
        H = torch.cat((H,X),1)
        # print(type(H))
        # print(type(X))
        # print( )
        # recurrent iteration of the network
        for i in range(self.n_iters):
            e = self.EdgeNet(H, Ri, Ro)
            H = self.NodeNet(H, e, Ri, Ro)
            # update H with the output of NodeNet
            H = torch.cat((H,X),1)
        # execute EdgeNet one more time to obtain edge predictions
        e = self.EdgeNet(H, Ri, Ro)
        # return edge prediction array
        return e
    
'''
###############################################################################
class GNN(tf.keras.Model):
    def __init__(self):
        #  Init function of GNN, inits all GNN blocks. 
        super(GNN, self).__init__(name='GNN')
        # Define Initial Input Layer
        self.InputNet =  tf.keras.layers.Dense(
            GNN.config['hid_dim'], input_shape=(3,),
            activation='relu',name='InputNet'
        )
        self.EdgeNet  = EdgeNet(name='EdgeNet')
        self.NodeNet  = NodeNet(name='NodeNet')
        self.n_iters  = GNN.config['n_iters']
    
    def call(self, graph_array):
        #  forward pass of the GNN 
        # decompose the graph array
        X, Ri, Ro = graph_array
        # execute InputNet to produce hidden dimensions
        H = self.InputNet(X)
        # add new dimensions to original X matrix
        H = tf.concat([H,X],axis=1)
        # recurrent iteration of the network
        for i in range(self.n_iters):
            e = self.EdgeNet(H, Ri, Ro)
            H = self.NodeNet(H, e, Ri, Ro)
            # update H with the output of NodeNet
            H = tf.concat([H,X],axis=1)
        # execute EdgeNet one more time to obtain edge predictions
        e = self.EdgeNet(H, Ri, Ro)
        # return edge prediction array
        return e
'''
