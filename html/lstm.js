function LSTM(W_cell,W_hidden,W_input,b) {
	var size = 100;

	var b_i = b.idx(0);
	var b_f = b.idx(1);
	var b_c = b.idx(2);
	var b_o = b.idx(3);

	var computeCell = NeuNet.applier(
		function(forget_gate,prev_cell,in_gate,cell_updates) {
			return forget_gate * prev_cell + in_gate * cell_updates;
		}
	);

	var addFour  = NeuNet.applier(function(x,h,b,c) { return x + h + b + c; });
	var addThree = NeuNet.applier(function(x,h,b,c) { return x + h + b; });

	return function(input,prev_hidden,prev_cell) {
		var x = input.dot(W_input);
		var h = prev_hidden.dot(W_hidden);
		var c = prev_cell.dot(W_cell);

		var x_i = x.slice(0 * size, 1 * size);
		var x_f = x.slice(1 * size, 2 * size);
		var x_c = x.slice(2 * size, 3 * size);
		var x_o = x.slice(3 * size, 4 * size);

		var h_i = h.slice(0 * size, 1 * size);
		var h_f = h.slice(1 * size, 2 * size);
		var h_c = h.slice(2 * size, 3 * size);
		var h_o = h.slice(3 * size, 4 * size);


		var c_i = c.slice(0 * size, 1 * size);
		var c_f = c.slice(1 * size, 2 * size);

		var in_lin     = addFour(x_i,h_i,b_i,c_i);
		var forget_lin = addFour(x_f,h_f,b_f,c_f);
		var cell_lin   = addThree(x_c,h_c,b_c);

		var in_gate      = NeuNet.sigmoid(in_lin);
		var forget_gate  = NeuNet.sigmoid(forget_lin);
		var cell_updates = NeuNet.tanh(cell_lin);

		var cell = computeCell(
				forget_gate,prev_cell,
				in_gate,cell_updates
			);

		var c_o = (cell.dot(W_cell)).slice(2 * size, 3 * size);
		var out_lin = addFour(x_o,h_o,b_o,c_o);
		var out_gate = NeuNet.sigmoid(out_lin);
		var hidden = out_gate.mult(NeuNet.tanh(cell))

		return { "hidden":hidden, "cell":cell };
	}
}
