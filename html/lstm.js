function LSTM(W_cell,W_hidden,W_input,b) {
	var size = 100;

	var b_i = b.idx(0);
	var b_f = b.idx(1);
	var b_c = b.idx(2);
	var b_o = b.idx(3);

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
		var in_lin     = x_i.plus(h_i).plus(b_i).plus(c_i);
		var forget_lin = x_f.plus(h_f).plus(b_f).plus(c_f);
		var cell_lin   = x_c.plus(h_c).plus(b_c);

		var in_gate      = NeuNet.sigmoid(in_lin);
		var forget_gate  = NeuNet.sigmoid(forget_lin);
		var cell_updates = NeuNet.tanh(cell_lin);

		var cell = (
				forget_gate.mult(prev_cell)
			).plus(
				in_gate.mult(cell_updates)
			);

		var c_o = (cell.dot(W_cell)).slice(2 * size, 3 * size);
		var out_lin = x_o.plus(h_o).plus(b_o).plus(c_o);
		var out_gate = NeuNet.sigmoid(out_lin);
		var hidden = out_gate.mult(NeuNet.tanh(cell))

		return { "hidden":hidden, "cell":cell };
	}
}
