package libnm.math.polynom;

import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class PolynomNewton {
	public PolynomNewton(Vector vecX, ExpTree expr) {
		m_vecX = vecX;
		m_expr = expr;
	}

	private double m_funcDiff(int i, int j) {
		if (i == j) {
			return m_expr.setVar("x", m_vecX.get(i)).calculate();
		} else {
			return (m_funcDiff(i, j - 1) - m_funcDiff(i + 1, j)) / (m_vecX.get(i) - m_vecX.get(j));
		}
	}

	public double getValue(double x) {
		double res = 0.0;

		for (int i = 0; i < m_vecX.getSize(); ++i) {
			double w = 1.0;

			for (int j = 0; j < i; ++j) {
				w *= x - m_vecX.get(j);
			}

			res += m_funcDiff(0, i) * w;
		}

		return res;
	}

	@Override
	public String toString() {
		String res = String.valueOf(m_funcDiff(0, 0));

		for (int i = 1; i < m_vecX.getSize(); ++i) {
			double f = m_funcDiff(0, i);

			if (f >= 0.0) {
				res += "+";
			}

			res += f;

			for (int j = 0; j < i; ++j) {
				res += "(x-" + m_vecX.get(j) + ")";
			}
		}

		return res;
	}

	private Vector m_vecX;
	private ExpTree m_expr;
}
