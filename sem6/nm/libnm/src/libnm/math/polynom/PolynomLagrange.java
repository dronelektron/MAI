package libnm.math.polynom;

import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class PolynomLagrange extends Polynom {
	public PolynomLagrange(Vector vecX, ExpTree expr) {
		super(vecX);

		m_vec = new Vector(vecX.getSize());

		Vector vecY = new Vector(vecX.getSize());
		Vector vecW = new Vector(vecX.getSize());

		for (int i = 0; i < vecX.getSize(); ++i) {
			set(i, vecX.get(i));
			vecY.set(i, expr.setVar("x", vecX.get(i)).calculate());
		}

		for (int i = 0; i < vecX.getSize(); ++i) {
			double w = 1.0;

			for (int j = 0; j < vecX.getSize(); ++j) {
				if (i != j) {
					w *= vecX.get(i) - vecX.get(j);
				}
			}

			vecW.set(i, w);
		}

		for (int i = 0; i < vecX.getSize(); ++i) {
			m_vec.set(i, vecY.get(i) / vecW.get(i));
		}
	}

	@Override
	public double getValue(double x) {
		double res = 0.0;

		for (int i = 0; i < getSize(); ++i) {
			double w = 1.0;

			for (int j = 0; j < getSize(); ++j) {
				if (i != j) {
					w *= x - get(j);
				}
			}

			res += m_vec.get(i) * w;
		}

		return res;
	}

	@Override
	public String toString() {
		String res = String.valueOf(m_vec.get(0));

		for (int i = 1; i < getSize(); ++i) {
			res += "(x-" + get(i) + ")";
		}

		for (int i = 1; i < getSize(); ++i) {
			if (m_vec.get(i) >= 0.0) {
				res += "+";
			}

			res += m_vec.get(i);

			for (int j = 0; j < getSize(); ++j) {
				if (i != j) {
					res += "(x-" + get(j) + ")";
				}
			}
		}

		return res;
	}

	private Vector m_vec;
}
