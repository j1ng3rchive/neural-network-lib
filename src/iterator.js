module.exports = Iterator;


class Iterator {
	constructor(size) {
		this.size = size;
		this._array = new Array(size).fill(undefined);
	}
	
	forEach(fn) {
		this._array.forEach((value, index) => fn(index));
	}

	map(fn) {
		return this._array.map((value, index) => fn(index));
	}

	findIndex(fn) {
		return this._array.findIndex((value, index) => fn(index));
	}

	find(fn) {
		return this._array.find((value, index) => fn(index));
	}

	reduce(fn) {
		return this._array.reduce((accumulator, value, index) => fn(accumulator, index));
	}

	fill(val) {
		return this._array.fill(val);
	}
}