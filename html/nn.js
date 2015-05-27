
NeuNet = (function() {
	var thresh = 10;
	function createArray(length) {
		var arr = new Array(length || 0),i = length;
		if (arguments.length > 1) {
			var args = Array.prototype.slice.call(arguments, 1);
			while(i--) arr[length-1 - i] = createArray.apply(this, args);
		} else {
			for (var i=0;i < arr.length; i++) arr[i] = 0.0;
		}
		return arr;
	}

	function transformApplier(fun) {
		return function(sym) {
			data = sym.data;
			data_res = createArray(sym.shape[0]);
			for (var i=0;i<data.length;i++) {
				data_res[i] = fun(data[i]);
				if (isNaN(data_res[i])) throw "NaN alert! " + data;
			}
			return new NeuNet(data_res);
		}
	}

	function binaryApplier(fun) {
		return function(sym1,sym2) {
			data1 = sym1.data;
			data2 = sym2.data;
			data_res = createArray(sym1.shape[0]);
			for (var i=0; i < sym1.shape[0]; i++) {
				data_res[i] = fun(data1[i],data2[i]);
				if (isNaN(data_res[i])) {
					console.log(sym1);
					console.log(sym2);
					throw "NaN alert!" + data1[i] + ", " + data2[i];
				}
			}
			return new NeuNet(data_res);
		}
	}


	function getDimensions(data) {
		if (data instanceof Array) {
			return [data.length].concat(getDimensions(data[0]));
		} else {
			return [];
		}
	}

	var NeuNet = function(data) {
		this.data = data;
		this.shape = getDimensions(data);
	}

	NeuNet.applier = function(fun) {
		var arg_count = fun.length;
		var buffer = new Array(arg_count);
		return function() {
			var result = new Array(arguments[0].shape[0]);
			for ( var i=0; i < result.length; i++ ) {
				for ( var j=0; j < arguments.length; j++ ) {
					buffer[j] = arguments[j].data[i];
				}
				result[i] = fun.apply(null,buffer);
			}
			return new NeuNet(result);
		}
	};

	NeuNet.sigmoid = transformApplier(function(x) {
		if ( x > thresh ) {
			return 1;
		} else if ( x < -thresh ) {
			return 0;
		} else {
			return 1 / (1 + Math.exp(-x))
		}
	});
	NeuNet.tanh = transformApplier(function(x) {
		if (x > thresh) {
			return 1;
		} else if ( x < -thresh) {
			return -1;
		} else {
			var x_ = Math.exp(2*x);
			return (x_ - 1)/(x_ + 1);
		}
	});
	NeuNet.neg     = transformApplier(function(x) { return -x; });
	NeuNet.plus    = binaryApplier(function(x,y) { return x + y; });
	NeuNet.mult    = binaryApplier(function(x,y) { return x * y; });
	NeuNet.sub     = binaryApplier(function(x,y) { return x - y; });
	NeuNet.div     = binaryApplier(function(x,y) { return x / y; });

	NeuNet.argmax  = function(vec) {
		var max_id = null;
		var max = -Infinity;
		for (var i = 0;i < vec.shape[0]; i++) {
			if (vec.data[i] > max) {
				max = vec.data[i];
				max_id = i;
			}
		}
		return max_id;
	};

	NeuNet.softmax = function(vec,temperature) {
		/**
		 * Avoids overflow issues by using something like the logsumexp trick.
		 * Divide both top and bottom by max, so it cancels out.
		 */
		temperature = temperature || 1;
		var max = vec.max();

		var result = vec.data.slice();
		var sum = 0;
		for (var i = 0;i < result.length; i++) {
			sum += result[i] = Math.exp((result[i] - max)/temperature);
		}

		for (var i = 0;i < result.length; i++) {
			result[i] = result[i] / sum;
		}
		return new NeuNet(result);
	};

	NeuNet.sample = function(vec) {
		var val = Math.random();
		var cum_prob = 0;
		for (var i = 0;i < vec.shape[0]; i++) {
			cum_prob += vec.data[i];
			if (cum_prob > val) {
				return i;
			}
		}
	};


	NeuNet.prototype = {
		dot: function(sym2) {
			var sym1 = this;
			if (sym1.shape[0] != sym2.shape[0]) throw "Dimensions wrong!";
			data1 = sym1.data;
			data2 = sym2.data;
			data_res = createArray(sym2.shape[1]);
			for (var j=0; j < data_res.length; j++) {
				var sum = 0;
				for (var i=0; i < sym1.shape[0]; i++) {
					sum += data1[i] * data2[i][j];
					if (isNaN(sum)) {
						throw "NaN alert! Sum so far: " + sum +
							" Value 1: " + data1[i] +
							" Value 2: " + data2[i][j];
					}
				}
				data_res[j] = sum;
			}
			return new NeuNet(data_res);
		},
		plus: function(sym2) { return NeuNet.plus(this,sym2); },
		mult: function(sym2) { return NeuNet.mult(this,sym2); },
		sub:  function(sym2) { return NeuNet.sub(this,sym2); },
		div:  function(sym2) { return NeuNet.div(this,sym2); },
		idx:  function(i) { return new NeuNet(this.data[i]); },
		slice: function(start,end) {
			return new NeuNet(this.data.slice(start,end));
		},
		max: function() { return Math.max.apply(null,this.data) }
	}

	return NeuNet;

})();
