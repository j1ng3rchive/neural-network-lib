import "iterator.js";
import "tensor.js";

class LinearNeuralNetwork {
	weights = null;
	biases = null;
	structure = null;
	neurons = null;

	constructor(structure) {
		this.initializeStructure(structure);
		this.initializeWeightsToOne();
		this.initializeBiasesToZero();
	}

	getDepth() {
		return this.structure.getDepth();
	}

	getSizeAtLayer(layer) {
		return this.structure.getSizeAtLayer(layer);
	}

	getNumberOfInputs() {
		return this.structure.getNumberOfInputs();
	}

	getNumberOfOutputs() {
		return this.structure.getNumberOfOutputs();
	}

	initializeWeightsToOne() {
		this.weights = new Tensor(
			this.getDepth() - 1,
			layer => this.getSizeAtLayer(layer),
			layer => this.getSizeAtLayer(layer + 1)
		).fill(1);
	}

	initializeBiasesToZero() {
		this.biases = new Tensor(
			this.getDepth() - 1,
			layer => this.getSizeAtLayer(layer)
		).fill(0);
	}

	initializeStructure(structure) {
		this.structure = LinearNeuralNetworkStructure.from(structure);
	}

	execute(tensorInput) {
		this.neurons = new Tensor(
			this.getDepth(),
			layer => neuralNetwork.getSizeAtLayer(layer)
		).fill(0);
		this.neurons.setIndex(tensorInput, 0);
		for(let layer = 1; layer < this.getDepth(); layer++) {
			let nextNeuralLayer = calculateNeuralLayer(layer);
			this.neurons.setIndex(nextNeuralLayer, layer);
		}
		return this.neurons.getIndex(this.getDepth() - 1);
	}

	calculateNeuralLayer(layer) {
		return this.weights.getIndex(layer - 1)
		.vectorMultiply(this.neurons.getIndex(layer - 1))
		.add(this.biases.getIndex(layer - 1))
		.RELU();
	}
}


class LinearNeuralNetworkStructure {
	depth = 0;
	sizeArray = [];

	constructor(sizeArray) {
		this.sizeArray = sizeArray;
		this.depth = sizeArray.length;
	}

	getDepth() {
		return this.depth;
	}

	getSizeAtLayer(layer) {
		return this.sizeArray[layer];
	}

	getNumberOfInputs() {
		return this.sizeArray[0];
	}

	getNumberOfOutputs() {
		return this.sizeArray[this.depth - 1];
	}

	mapSizes(fn) {
		return this.sizeArray.map(fn);
	}

	static from(structure) {
		if(structure instanceof LinearNeuralNetworkStructure) {
			return structure;
		} else {
			return LinearNeuralNetworkStructure(structure);
		}
	}
}