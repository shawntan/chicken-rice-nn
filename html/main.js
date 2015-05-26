$.getJSON("params.json",function(params) {
	var P = {};

	for (var k in params) { P[k] = new NeuNet(params[k]); }

	lstm_1 = LSTM(
			P.W_recurrent_1_cell,
			P.W_recurrent_1_hidden,
			P.W_recurrent_1_input,
			P.b_recurrent_1
		);

	lstm_2 = LSTM(
			P.W_recurrent_2_cell,
			P.W_recurrent_2_hidden,
			P.W_recurrent_2_input,
			P.b_recurrent_2
		);

	var next_word = function(c,h1,c1,h2,c2) {
		word_vec = P.V.idx(c);
		layer_1 = lstm_1(word_vec,h1,c1);
		layer_2 = lstm_2(layer_1.hidden,h2,c2);
		output = NeuNet.softmax((layer_2.hidden.dot(P.W_output)).plus(P.b_output));
		return {
			"output":output,
			"h1": layer_1.hidden,
			"c1": layer_1.cell,
			"h2": layer_2.hidden,
			"c2": layer_2.cell,
		};
	}

	var model = {
		"next_word":next_word,
		"init_h1":NeuNet.tanh(P.init_recurrent_1_hidden),
		"init_h2":NeuNet.tanh(P.init_recurrent_2_hidden),
		"init_c1":P.init_recurrent_1_cell,
		"init_c2":P.init_recurrent_2_cell,
		"vocab": [
			'\n', ' ', '!', '"', '#', '$', '%', '&', "'",
			'(', ')', '*', '+', ',', '-', '.', '/', '0',
			'1', '2', '3', '4', '5', '6', '7', '8', '9',
			':', ';', '<', '=', '>', '?', '@', 'A', 'B',
			'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
			'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
			'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']',
			'^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f',
			'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
			'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
			'y', 'z', '{', '|', '}', '~'
		]
	};
	sample = function(prime) {
		console.log(prime);
		var result = prime;
		var start_id = model.vocab.length;
		var state = model.next_word(start_id,model.init_h1,model.init_c1,model.init_h2,model.init_c2);

		for (var i=0;i < prime.length; i++) {
			var id = model.vocab.indexOf(prime[i]);
			state = model.next_word(id,state.h1,state.c1,state.h2,state.c2);
		}

		var id = NeuNet.sample(state.output);
		while((model.vocab[id] != "\n") && (result.length < 200)) {
			result += model.vocab[id];
			state = model.next_word(id,state.h1,state.c1,state.h2,state.c2);
			id = NeuNet.sample(state.output);
		}
		return result;
	}
	function hashString() {
		var hash = document.location.hash;
		if (hash && hash.length > 0) {
			return hash.substring(1,hash.length);
		} else {
			return "";
		}
	}

	var textbox = $("#gentext")
	textbox.text(sample(hashString()));
	setInterval(function() {
		textbox.text(sample(hashString()));
	},5000);
});
