package main;

import libnm.math.Vector;
import libnm.util.Logger;

public class Euler {
	public Euler(double a1, double a2, double a3, double y0, double a, double b, double h) {
		m_a1 = a1;
		m_a2 = a2;
		m_a3 = a3;
		m_y0 = y0;
		m_a = a;
		m_b = b;
		m_h = h;
		m_logger = null;
	}

	public void setLogger(Logger logger) {
		m_logger = logger;
	}

	public void solve(Vector vecX, Vector vecY) {
		int n = (int)((m_b - m_a) / m_h) + 1;
		int maxInd = 0;
		int minInd = 0;
		double maxValue = m_y0;
		double minValue = m_y0;

		vecX.set(0, m_a);
		vecY.set(0, m_y0);

		for (int i = 1; i < n; ++i) {
			double x = vecX.get(i - 1);
			double y = vecY.get(i - 1);
			double xkh = x + 0.5 * m_h;
			double ykh = y + 0.5 * m_h * f(vecX, vecY, y, x - m_a3);
			double yk = y + m_h * f(vecX, vecY, ykh, xkh - m_a3);

			vecX.set(i, x + m_h);
			vecY.set(i, yk);

			if (vecY.get(i) > maxValue) {
				maxValue = vecY.get(i);
				maxInd = i;
			}

			if (vecY.get(i) < minValue) {
				minValue = vecY.get(i);
				minInd = i;
			}
		}

		if (m_logger != null) {
			m_logger.writeln("Минимум функции в точке x = " + vecX.get(minInd) + ", y = " + minValue);
			m_logger.writeln("Максимум функции в точке x = " + vecX.get(maxInd) + ", y = " + maxValue);
			m_logger.writeln("Расстояние между экстремумами: " + Math.abs(vecX.get(maxInd) - vecX.get(minInd)));
		}
	}

	private double f(Vector vecX, Vector vecY, double y, double offset) {
		int n = vecX.getSize();
		int ind = -1;
		double yFinal;

		for (int i = 0; i < n; ++i) {
			if (vecX.get(i) > offset) {
				break;
			}

			ind = i;
		}

		if (offset < m_a) {
			yFinal = m_y0;
		} else if (vecX.get(ind) == offset) {
			yFinal = vecY.get(ind);
		} else if (ind > 0 && ind < n - 1){
			Vector tmpVecX = new Vector(new double[] {vecX.get(ind - 1), vecX.get(ind), vecX.get(ind + 1)});
			Vector tmpVecY = new Vector(new double[] {vecY.get(ind - 1), vecY.get(ind), vecY.get(ind + 1)});
			PolynomLagrange poly = new PolynomLagrange(tmpVecX, tmpVecY);

			yFinal = poly.getValue(offset);
		} else {
			yFinal = m_y0;
		}

		return m_a2 - m_a1 * y - Math.pow(yFinal, 2.0);
	}

	private double m_a1;
	private double m_a2;
	private double m_a3;
	private double m_y0;
	private double m_a;
	private double m_b;
	private double m_h;
	private Logger m_logger;
}
