let Tensor = require("iterator.js");
let Iterator = require("tensor.js");

class LinearNeuralNetwork {
	constructor(structure) {
		this.neurons = null;
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
		this.structure = LinearNeuralNetwork.Structure.from(structure);
	}

	execute(tensorInput) {
		this.neurons = new Tensor(
			this.getDepth(),
			layer => this.neuralNetwork.getSizeAtLayer(layer)
		).fill(0);
		this.neurons.setIndex(tensorInput, 0);

		//We skip 0 because the 0th layer is already defined
		for(let layer = 1; layer < this.getDepth(); layer++) {
			let nextNeuralLayer = this.calculateNeuralLayer(layer);
			this.neurons.setIndex(nextNeuralLayer, layer);
		}
	}

	calculateNeuralLayer(layer) {
		return this.weights.getIndex(layer - 1)
			.vectorMultiply(this.neurons.getIndex(layer - 1))
			.add(this.biases.getIndex(layer - 1))
			.RELU();
	}

	calculateWeightGradient(tensorInput) {
		//TODO
	}

	calculateWeightDifferentialOfInputForPath(tensorInput, path) {
		//TODO
	}
}


LinearNeuralNetwork.Structure = class LinearNeuralNetworkStructure {
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
		if(structure instanceof LinearNeuralNetwork.Structure) {
			return structure;
		} else {
			return LinearNeuralNetwork.Structure(structure);
		}
	}
}

module.exports = LinearNeuralNetwork;