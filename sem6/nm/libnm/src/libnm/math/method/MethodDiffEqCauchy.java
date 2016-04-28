package libnm.math.method;

import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class MethodDiffEqCauchy {
	public MethodDiffEqCauchy(ExpTree exprP, ExpTree exprQ, ExpTree exprF, double y0, double z0, double a, double b, double h) {
		m_exprP = exprP;
		m_exprQ = exprQ;
		m_exprF = exprF;
		m_y0 = y0;
		m_z0 = z0;
		m_a = a;
		m_b = b;
		m_h = h;
	}

	public void euler(Vector vecX, Vector vecY) {
		int n = getN();
		Vector vecZ = new Vector(n);

		vecX.set(0, m_a);
		vecY.set(0, m_y0);
		vecZ.set(0, m_z0);

		for (int i = 1; i < n; ++i) {
			vecX.set(i, vecX.get(i - 1) + m_h);
			vecY.set(i, vecY.get(i - 1) + m_h * vecZ.get(i - 1));
			vecZ.set(i, vecZ.get(i - 1) + m_h * g(vecX.get(i - 1), vecY.get(i - 1), vecZ.get(i - 1)));
		}
	}

	public void rungeKutta(Vector vecX, Vector vecY, Vector vecZ) {
		int n = getN();

		vecX.set(0, m_a);
		vecY.set(0, m_y0);
		vecZ.set(0, m_z0);

		for (int i = 1; i < n; ++i) {
			vecX.set(i, vecX.get(i - 1) + m_h);
			vecY.set(i, vecY.get(i - 1) + dy(vecX.get(i - 1), vecY.get(i - 1), vecZ.get(i - 1)));
			vecZ.set(i, vecZ.get(i - 1) + dz(vecX.get(i - 1), vecY.get(i - 1), vecZ.get(i - 1)));
		}
	}

	public void adams(Vector vecX, Vector vecY) {
		int n = getN();
		double b = m_b;
		Vector vecZ = new Vector(n);

		setB(m_a + 3.0 * m_h);
		rungeKutta(vecX, vecY, vecZ);
		setB(b);

		for (int i = 4; i < n; ++i) {
			double deltaY = 0.0;
			double deltaZ = 0.0;

			deltaY += 55.0 * vecZ.get(i - 1);
			deltaY -= 59.0 * vecZ.get(i - 2);
			deltaY += 37.0 * vecZ.get(i - 3);
			deltaY -= 9.0 * vecZ.get(i - 4);
			deltaY /= 24.0;

			deltaZ += 55.0 * g(vecX.get(i - 1), vecY.get(i - 1), vecZ.get(i - 1));
			deltaZ -= 59.0 * g(vecX.get(i - 2), vecY.get(i - 2), vecZ.get(i - 2));
			deltaZ += 37.0 * g(vecX.get(i - 3), vecY.get(i - 3), vecZ.get(i - 3));
			deltaZ -= 9.0 * g(vecX.get(i - 4), vecY.get(i - 4), vecZ.get(i - 4));
			deltaZ /= 24.0;

			vecX.set(i, vecX.get(i - 1) + m_h);
			vecY.set(i, vecY.get(i - 1) + m_h * deltaY);
			vecZ.set(i, vecZ.get(i - 1) + m_h * deltaZ);
		}
	}

	public int getN() {
		return (int)((m_b - m_a) / m_h) + 1;
	}

	private void setB(double b) {
		m_b = b;
	}

	public void setH(double h) {
		m_h = h;
	}

	private double g(double x, double y, double z) {
		m_exprP.setVar("x", x);
		m_exprP.setVar("y", y);
		m_exprP.setVar("z", z);
		m_exprQ.setVar("x", x);
		m_exprQ.setVar("y", y);
		m_exprQ.setVar("z", z);
		m_exprF.setVar("x", x);
		m_exprF.setVar("y", y);
		m_exprF.setVar("z", z);

		return -(m_exprP.calculate() * z + m_exprQ.calculate() * y + m_exprF.calculate());
	}

	private double k1(double z) {
		return m_h * z;
	}

	private double l1(double x, double y, double z) {
		return m_h * g(x, y, z);
	}

	private double k2(double x, double y, double z) {
		return m_h * (z + 0.5 * l1(x, y, z));
	}

	private double l2(double x, double y, double z) {
		return m_h * g(x + 0.5 * m_h, y + 0.5 * k1(z), z + 0.5 * l1(x, y, z));
	}

	private double k3(double x, double y, double z) {
		return m_h * (z + 0.5 * l2(x, y, z));
	}

	private double l3(double x, double y, double z) {
		return m_h * g(x + 0.5 * m_h, y + 0.5 * k2(x, y, z), z + 0.5 * l2(x, y, z));
	}

	private double k4(double x, double y, double z) {
		return m_h * (z + l3(x, y, z));
	}

	private double l4(double x, double y, double z) {
		return m_h * g(x + m_h, y + k3(x, y, z), z + l3(x, y, z));
	}

	private double dy(double x, double y, double z) {
		return (k1(z) + 2.0 * k2(x, y, z) + 2.0 * k3(x, y, z) + k4(x, y, z)) / 6.0;
	}

	private double dz(double x, double y, double z) {
		return (l1(x, y, z) + 2.0 * l2(x, y, z) + 2.0 * l3(x, y, z) + l4(x, y, z)) / 6.0;
	}

	private ExpTree m_exprP;
	private ExpTree m_exprQ;
	private ExpTree m_exprF;
	private double m_y0;
	private double m_z0;
	private double m_a;
	private double m_b;
	private double m_h;
}
