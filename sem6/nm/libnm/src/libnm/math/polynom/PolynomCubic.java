package libnm.math.polynom;

import libnm.math.Matrix;
import libnm.math.method.MethodSle;
import libnm.math.Vector;

public class PolynomCubic extends Polynom {
	public PolynomCubic(Vector vecX, Vector vecY) {
		super(vecX);

		int n = getSize();
		Matrix mat = new Matrix(n - 2);
		Vector vec = new Vector(n - 2);
		MethodSle method = new MethodSle();
		Vector vecH = new Vector(n - 1);
		
		m_vecA = new Vector(n - 1);
		m_vecB = new Vector(n - 1);
		m_vecC = new Vector(n - 1);
		m_vecD = new Vector(n - 1);

		for (int i = 0; i < n - 1; ++i) {
			vecH.set(i, get(i + 1) - get(i));
		}

		for (int i = 0; i < n - 2; ++i) {
			vec.set(i, 3.0 * ((vecY.get(i + 2) - vecY.get(i + 1)) / vecH.get(i) - (vecY.get(i + 1) - vecY.get(i)) / vecH.get(i + 1)));
		}

		mat.set(0, 0, 2.0 * (vecH.get(0) + vecH.get(1)));
		mat.set(0, 1, vecH.get(1));

		for (int i = 1; i < n - 3; ++i) {
			mat.set(i, i - 1, vecH.get(i - 1));
			mat.set(i, i, 2.0 * (vecH.get(i - 1) + vecH.get(i)));
			mat.set(i, i + 1, vecH.get(i));
		}

		mat.set(n - 3, n - 4, vecH.get(n - 3));
		mat.set(n - 3, n - 3, 2.0 * (vecH.get(n - 4) + vecH.get(n - 3)));

		method.tma(mat, vec, m_vecC);

		for (int i = n - 2; i > 0; --i) {
			m_vecC.set(i, m_vecC.get(i - 1));
		}

		m_vecC.set(0, 0.0);

		for (int i = 0; i < n - 1; ++i) {
			m_vecA.set(i, vecY.get(i));
		}

		for (int i = 0; i < n - 2; ++i) {
			double h = vecH.get(i);

			m_vecB.set(i, (vecY.get(i + 1) - vecY.get(i)) / h - h * (m_vecC.get(i + 1) + 2.0 * m_vecC.get(i)) / 3.0);
			m_vecD.set(i, (m_vecC.get(i + 1) - m_vecC.get(i)) / (3.0 * h));
		}

		double h = vecH.get(n - 2);

		m_vecB.set(n - 2, (vecY.get(n - 1) - vecY.get(n - 2)) / h - 2.0 * h * m_vecC.get(n - 2) / 3.0);
		m_vecD.set(n - 2, -m_vecC.get(n - 2) / (3.0 * h));
	}

	@Override
	public double getValue(double x) {
		for (int i = 0; i < getSize() - 1; ++i) {
			if (get(i) <= x && x <= get(i + 1)) {
				double h = x - get(i);

				return m_vecA.get(i) + m_vecB.get(i) * h + m_vecC.get(i) * Math.pow(h, 2.0) + m_vecD.get(i) * Math.pow(h, 3.0);
			}
		}

		return 0.0;
	}

	@Override
	public String toString() {
		String res = "";

		for (int i = 0; i < getSize() - 1; ++i) {
			String strH = "(x-" + get(i) + ")";

			res += "S(" + (i + 1) + ")=";
			res += m_vecA.get(i);

			if (m_vecB.get(i) >= 0.0) {
				res += "+";
			}

			res += m_vecB.get(i) + strH + "^1";

			if (m_vecC.get(i) >= 0.0) {
				res += "+";
			}

			res += m_vecC.get(i) + strH + "^2";

			if (m_vecD.get(i) >= 0.0) {
				res += "+";
			}

			res += m_vecD.get(i) + strH + "^3";

			if (i + 1 < getSize() - 1) {
				res += "\n";
			}
		}

		return res;
	}

	private Vector m_vecA;
	private Vector m_vecB;
	private Vector m_vecC;
	private Vector m_vecD;
}
