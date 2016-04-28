package libnm.math.method;

import libnm.math.expression.ExpTree;

public class MethodIntegral {
	public MethodIntegral(ExpTree expr, double x0, double x1, double h) {
		m_expr = expr;
		m_x0 = x0;
		m_x1 = x1;
		m_h = h;
	}

	public double rectangle() {
		double res = 0.0;

		for (double x = m_x0 + m_h; x <= m_x1; x += m_h) {
			res += m_expr.setVar("x", (2.0 * x - m_h) / 2.0).calculate();
		}

		return res * m_h;
	}

	public double trapezoidal() {
		double res = m_expr.setVar("x", m_x0).calculate() / 2.0;

		for (double x = m_x0 + m_h; x <= m_x1 - m_h; x += m_h) {
			res += m_expr.setVar("x", x).calculate();
		}

		res += m_expr.setVar("x", m_x1).calculate() / 2.0;

		return res * m_h;
	}

	public double simpson() {
		double res = m_expr.setVar("x", m_x0).calculate();
		boolean isTwo = false;

		for (double x = m_x0 + m_h; x <= m_x1 - m_h; x += m_h) {
			res += m_expr.setVar("x", x).calculate() * (isTwo ? 2.0 : 4.0);
			isTwo = !isTwo;
		}

		res += m_expr.setVar("x", m_x1).calculate();

		return res * m_h / 3.0;
	}

	public void setH(double h) {
		m_h = h;
	}

	private ExpTree m_expr;
	private double m_x0;
	private double m_x1;
	private double m_h;
}
