package main;

import libnm.math.Vector;

public class PolynomLagrange {
	public PolynomLagrange(Vector vecX, Vector vecY) {
		Vector vecW = new Vector(vecX.getSize());

		m_vecX = vecX;
		m_vecY = new Vector(vecX.getSize());
		m_vecY.copy(vecY);

		for (int i = 0; i < m_vecX.getSize(); ++i) {
			double w = 1.0;

			for (int j = 0; j < m_vecX.getSize(); ++j) {
				if (i != j) {
					w *= m_vecX.get(i) - m_vecX.get(j);
				}
			}

			vecW.set(i, w);
		}

		for (int i = 0; i < m_vecX.getSize(); ++i) {
			m_vecY.set(i, m_vecY.get(i) / vecW.get(i));
		}
	}

	public double getValue(double x) {
		int n = m_vecX.getSize();
		double res = 0.0;

		for (int i = 0; i < n; ++i) {
			double w = 1.0;

			for (int j = 0; j < n; ++j) {
				if (i != j) {
					w *= x - m_vecX.get(j);
				}
			}

			res += m_vecY.get(i) * w;
		}

		return res;
	}

	@Override
	public String toString() {
		int n = m_vecX.getSize();
		String res = String.valueOf(m_vecY.get(0));

		for (int i = 1; i < n; ++i) {
			res += "(x-" + m_vecX.get(i) + ")";
		}

		for (int i = 1; i < n; ++i) {
			if (m_vecY.get(i) >= 0.0) {
				res += "+";
			}

			res += m_vecY.get(i);

			for (int j = 0; j < n; ++j) {
				if (i != j) {
					res += "(x-" + m_vecX.get(j) + ")";
				}
			}
		}

		return res;
	}

	private Vector m_vecX;
	private Vector m_vecY;
}
