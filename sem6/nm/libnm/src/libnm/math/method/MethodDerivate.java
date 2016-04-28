package libnm.math.method;

import libnm.math.Vector;

public class MethodDerivate {
	public MethodDerivate(Vector vecX, Vector vecY, double x) {
		m_vecX = vecX;
		m_vecY = vecY;
		m_x = x;
	}

	public double deriv1() {
		int i = 0;
		double res = 0.0;

		for (int j = 0; j < m_vecX.getSize(); ++j) {
			if (m_vecX.get(j) == m_x) {
				i = j;

				break;
			}
		}

		res += (m_vecY.get(i + 1) - m_vecY.get(i)) / (m_vecX.get(i + 1) - m_vecX.get(i));
		res -= (m_vecY.get(i) - m_vecY.get(i - 1)) / (m_vecX.get(i) - m_vecX.get(i - 1));
		res /= m_vecX.get(i + 1) - m_vecX.get(i - 1);
		res *= (2.0 * m_x - m_vecX.get(i - 1) - m_vecX. get(i));
		res += (m_vecY.get(i) - m_vecY.get(i - 1)) / (m_vecX.get(i) - m_vecX.get(i - 1));

		return res;
	}

	public double deriv2() {
		int i = 0;
		double res = 0.0;

		for (int j = 0; j < m_vecX.getSize(); ++j) {
			if (m_vecX.get(j) == m_x) {
				i = j;

				break;
			}
		}

		res += (m_vecY.get(i + 1) - m_vecY.get(i)) / (m_vecX.get(i + 1) - m_vecX.get(i));
		res -= (m_vecY.get(i) - m_vecY.get(i - 1)) / (m_vecX.get(i) - m_vecX.get(i - 1));
		res /= m_vecX.get(i + 1) - m_vecX.get(i - 1);
		res *= 2.0;

		return res;
	}

	private Vector m_vecX;
	private Vector m_vecY;
	private double m_x;
}
