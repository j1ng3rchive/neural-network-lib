import "iterator.js";

module.exports = {
	TensorPath,
	Tensor
};


class TensorPath {
	path = [];

	constructor(path = []) {
		this.path = path;
		this.setDimension(path.length);
	}

	setDimension(dimension) {
		this.dimension = dimension;
		this.iterator = new Iterator(dimension);
	}

	getDimension() {
		return this.dimension;
	}

	lambda(fn) {
		return index => fn(this.path[index], index, this);
	}

	forEach(fn) {
		this.iterator.forEach(this.lambda(fn));
	}

	map(fn) {
		return this.iterator.map(this.lambda(fn));
	}

	findIndex(fn) {
		return this.iterator.findIndex(this.lambda(fn));
	}

	find(fn) {
		return this.iterator.find(this.lambda(fn));
	}

	reduce(fn, startValue) {
		return this.iterator.reduce(this.lambda(fn), startValue);
	}

	fill(val) {
		this.path.fill(val);
		return this;
	}

	getComponent(index) {
		return this.path[index];
	}

	setComponent(dimension, index) {
		this.path[dimension] = index;
	}
}


class Tensor {
	sizes = [];
	subtensors = null;
	dimension = 0;

	constructor(...sizes) {
		this.sizes = sizes.flatten();
		this.dimension = sizes.length;
	}

	fill(value) {
		if(this.isScalar()) {
			this.subtensors = value;
		} else {
			this.subtensors = new Iterator(sizes[0]).map(index =>
				new Tensor(this.getSubSizes(index)).initializeWithIdenticalsubtensors(value)
			);
		}
		return this;
	}

	setAll(subtensors) {
		this.subtensors = subtensors;
		return this;
	}

	getIndex(index) {
		return this.subtensors[index];
	}

	setIndex(subtensor, index) {
		this.subtensors[index] = subtensor;
	}

	getPath(path) {
		return path.reduce((subtensor, index) => subtensor.getIndex(index), this);
	}

	isVector() {
		return this.dimension == 1;
	}

	isScalar() {
		return this.dimension == 0;
	}

	getScalarValue() {
		return this.subtensors;
	}

	getSubSizes(index) {
		return sizes.slice(1).map(Tensor.getSubSize);
	}

	forEach(fn) {
		return this.forEachRecursive(fn, new TensorPath().setDimension(this.getDimension()), 0, this);
	}

	forEachRecursive(fn, path, dimension, supertensor) {
		if(this.isScalar()) {
			fn(this.getScalarValue(), path, supertensor);
		}
		this.subtensors.forEach((subtensor, index) =>
			subtensor.forEachRecursive(fn, path.setComponent(dimension, index), dimension + 1, supertensor)
		);
	}
	
	shallowMap(fn) {
		let mappedSubtensors = this.subtensors.map((subtensor, index) => fn(subtensor, index, this));
		return new Tensor(this.sizes).setAll(mappedSubtensors);
	}

	map(fn) {
		return this.mapRecursive(fn, new TensorPath().setDimension(this.getDimension()), 0, this);
	}

	mapRecursive(fn, path, dimension, supertensor) {
		if(this.isScalar()) {
			return fn(this.getScalarValue(), path, supertensor);
		}
		let mappedSubtensors = this.subtensors.map((subtensor, index) =>
			subtensor.mapRecursive(fn, path.setComponent(dimension, index), dimension + 1, supertensor)
		);
		return new Tensor(this.sizes).setAll(mappedSubtensors);
	}

	some(fn) {
		return this.someRecursive(fn, new TensorPath().setDimension(this.getDimension()), 0, this);
	}

	someRecursive(fn, path, dimension, supertensor) {
		if(this.isScalar()) {
			return !!fn(this.getScalarValue(), path, supertensor);
		} else {
			return this.subtensors.some((subtensor, index) =>
				subtensor.someRecursive(fn, path.setComponent(dimension, index), dimension + 1, supertensor)
			);
		}
	}

	findPath(fn) {
		let foundPath = new TensorPath().setDimension(this.getDimension());
		let found = this.someRecursive(fn, foundPath, 0, this);
		if(found) {
			return foundPath;
		} else {
			return -1;
		}
	}

	findTensor(fn) {
		let foundPath = new TensorPath().setDimension(this.getDimension());
		let found = this.someRecursive(fn, foundPath, 0, this);
		if(found) {
			return this.getPath(foundPath);
		} else {
			return undefined;
		}
	}

	add(tensor) {
		return this.map((value, path) => value + tensor.getPath(path));
	}

	dot(vector) {
		return this.subtensors.reduce((sum, subtensor) =>
			sum + subtensor.getScalarValue() * vector.getScalarValue()
		, 0);
	}

	vectorMultiply(vector) {
		return this.vectorMultiplyRecursive(vector, new TensorPath().setDimension(this.getDimension() - 1), 0);
	}

	vectorMultiplyRecursive(vector, path, dimension) {
		if(this.isVector()) {
			return this.dot(vector);
		}
		let mappedSubtensors = this.subtensors.map((subtensor, index) =>
			subtensor.vectorMultiplyRecursive(vector, path.setComponent(dimension, index), dimension + 1)
		);
		return new Tensor(this.sizes.slice(0,-1)).setAll(mappedSubtensors);
	}

	RELU() {
		return this.map(Tensor.RELU);
	}

	UnitStep() {
		return this.map(Tensor.UnitStep);
	}

	static getSubSize(size, index) {
		if(size instanceof Function) {
			return Tensor.activateSizeFnWithIndex(size, index);
		} else {
			return size;
		}
	}

	static activateSizeFnWithIndex(size, index) {
		if(index == 0) {
			return size(index);
		} else {
			return (...args) => size(index, ...args);
		}
	}

	static RELU(x) {
		return Math.max(x, 0);
	}

	static UnitStep(x) {
		return +(x > 0);
	}
}