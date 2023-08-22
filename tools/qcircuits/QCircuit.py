import cirq
import json
import numpy as np
import qcircuits.circuits as qcircuits
from qiskit import QuantumCircuit

class QCircuit:
	def __init__(self, IEC_id, PQC_id, MC_id, n_layers=1, input_size=4, p=None):
		self.n_layers = n_layers
		self.n_inputs = input_size

		self.IEC_id = IEC_id
		self.PQC_id = PQC_id
		self.MC_id  = MC_id
		self.p = p
		
		# read metadata
		with open('qcircuits/circuits_metadata.json') as json_file:
			self.metadata = json.load(json_file)

		self.n_qubits = self.get_n_qubits()
		self.n_params = self.get_n_params()
		self.n_measurements = self.get_measurements()

	def model_circuit(self):
		# self.qubits  = cirq.GridQubit.rect(self.n_qubits, 1)
		# self.circuit = cirq.Circuit()
		self.IEC_circuit = QuantumCircuit(self.n_qubits)

		self.IEC()(self.IEC_circuit, n_qubits = self.n_qubits)
		# self.IEC()(self.IEC_circuit, n_qubits = self.n_qubits)

		self.PQC_circuit = QuantumCircuit(self.n_qubits)
		self.PQC()(self.PQC_circuit, n_layers = self.n_layers, n_qubits = self.n_qubits)
		# self.PQC()(self.PQC_circuit, n_layers = self.n_layers, n_qubits = self.n_qubits)
		
		self.circuit=QuantumCircuit(self.n_qubits)
		self.circuit.compose(self.IEC_circuit,inplace=True)
		self.circuit.compose(self.PQC_circuit,inplace=True)
		

		# return self.circuit, self.qubits
		return self.circuit, self.IEC_circuit.parameters, self.PQC_circuit.parameters


	def pqc_circuit(self):
		# self.qubits  = cirq.GridQubit.rect(self.n_qubits, 1)
		# self.circuit = cirq.Circuit()
		self.circuit = QuantumCircuit(self.n_qubits)


		self.PQC()(self.circuit, n_layers = self.n_layers, n_qubits = self.n_qubits)
		# self.PQC()(self.circuit, self.qubits, n_layers = self.n_layers, n_qubits = self.n_qubits)
		
		# return self.circuit, self.qubits
		return self.circuit

	def IEC(self):
		'''information encoding circuit'''
		return getattr(qcircuits, self.metadata['qc_iec_dict'][self.IEC_id])
	def PQC(self):
		'''parametrized quantum circuit'''
		return getattr(qcircuits, self.metadata['qc_pqc_dict'][self.PQC_id])
	def measurement_operators(self):
		'''measurement block of the circuit'''
		
		self.qc_meas_dict = {
			'measure_all': qcircuits.measure_all,
			'measure_last': qcircuits.measure_last,
			'measure_TNN': qcircuits.measure_TNN,
			'measure_TNN_qiskit': qcircuits.measure_TNN_qiskit,
			#'none': qcircuits.state_vector,
			#'probs': qcircuits.probs,
			#'samples': qcircuits.sample,
			}
		return self.qc_meas_dict[self.MC_id](n_measurements=self.n_measurements)
		

	def get_n_params(self):
		if self.PQC_id in self.metadata['weight_shapes_dict'].keys():
			n_params  = self.metadata['weight_shapes_dict'][self.PQC_id]*self.n_layers
		elif self.PQC_id == 'TNN_con3_qiskit':
			n_params =  12*self.n_layers
		elif self.PQC_id == 'TNN_con3_qiskit_6':
			n_params =  12*self.n_layers
		elif self.PQC_id == 'TNN_con3_qiskit_test':
			n_params =  self.n_qubits*self.n_layers
		elif self.PQC_id == '19':
			n_params =  3*self.n_qubits*self.n_layers
		elif self.PQC_id == '15':
			n_params =  2*self.n_qubits*self.n_layers
		elif self.PQC_id == '14':
			n_params = (3*self.n_qubits + self.n_qubits/np.gcd(self.n_qubits,3))*self.n_layers
		elif self.PQC_id == '10':
			n_params = self.n_qubits*(self.n_layers+1)
		elif self.PQC_id == '10P':
			n_params = 2*self.n_qubits*(self.n_layers+1)
		elif self.PQC_id == '7':
			n_params = (5*self.n_qubits-1)*self.n_layers
		elif self.PQC_id == '6':
			n_params = (self.n_qubits**2 + 3*self.n_qubits)*self.n_layers
		elif self.PQC_id == '3':
			n_params = (3*self.n_qubits-1)*self.n_layers
		elif self.PQC_id == 'generic':
			n_params = (6*self.n_qubits)*self.n_layers
		elif self.PQC_id == 'TTN':
			n_params = int(2**(np.log2(self.n_qubits)+1)-2 +1)
		elif self.PQC_id == 'MPS':
			n_params = 2*self.n_qubits - 1
		elif self.PQC_id == '10_local':
			n_params = self.n_qubits*self.n_layers+1
		elif self.PQC_id == '10_2des':
			n_params = self.n_qubits*(self.n_layers+1)
		elif self.PQC_id == '10_2':
			n_params = self.n_qubits*(self.n_layers+1)
		elif self.PQC_id == '10_3':
			n_params = self.n_qubits*(self.n_layers+1)
		elif self.PQC_id == '10_identity':
			n_params = self.n_qubits*(self.n_layers+1)*2
		else:
			raise ValueError('PQC weights not defined')
		return n_params


	def get_n_qubits(self):
		# set number of qubits and inputs
		if self.IEC_id in self.metadata['n_qubits_dict'].keys():
			n_qubits = self.metadata['n_qubits_dict'][self.IEC_id]
		else:
			n_qubits = self.n_inputs
		return n_qubits

	def get_measurements(self):
		# set number of measurements
		n_measurements_dict = {
			'measure_all': self.n_qubits,
			'measure_last': 1,
			'measure_TNN': 3,
			'measure_TNN_qiskit': 3,
			#'none': 0,
			#'probs': self.n_qubits**2,
			#'samples': 1000*(self.n_qubits**2),
			}
		return n_measurements_dict[self.MC_id]
