package libnm.math.polynom;

import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class PolynomNewton extends Polynom {
	public PolynomNewton(Vector vec, ExpTree expr) {
		super(vec);

		m_expr = expr;
	}

	private double m_funcDiff(int i, int j) {
		if (i == j) {
			return m_expr.setVar("x", get(i)).calculate();
		} else {
			return (m_funcDiff(i, j - 1) - m_funcDiff(i + 1, j)) / (get(i) - get(j));
		}
	}

	@Override
	public double getValue(double x) {
		double res = 0.0;

		for (int i = 0; i < getSize(); ++i) {
			double w = 1.0;

			for (int j = 0; j < i; ++j) {
				w *= x - get(j);
			}

			res += m_funcDiff(0, i) * w;
		}

		return res;
	}

	@Override
	public String toString() {
		String res = String.valueOf(m_funcDiff(0, 0));

		for (int i = 1; i < getSize(); ++i) {
			double f = m_funcDiff(0, i);

			if (f >= 0.0) {
				res += "+";
			}

			res += f;

			for (int j = 0; j < i; ++j) {
				res += "(x-" + get(j) + ")";
			}
		}

		return res;
	}

	private ExpTree m_expr;
}
