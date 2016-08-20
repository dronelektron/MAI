package libnm.math.pde;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class Parabolic {
	public void setA(double a) {
		m_a = a;
	}

	public void setB(double b) {
		m_b = b;
	}

	public void setC(double c) {
		m_c = c;
	}

	public void setF(ExpTree f) {
		m_f = f;
	}

	public void setPsi(ExpTree psi) {
		m_psi = psi;
	}

	public void setAlpha(double alpha) {
		m_alpha = alpha;
	}

	public void setBeta(double beta) {
		m_beta = beta;
	}

	public void setGamma(double gamma) {
		m_gamma = gamma;
	}

	public void setDelta(double delta) {
		m_delta = delta;
	}

	public void setFi0(ExpTree fi0) {
		m_fi0 = fi0;
	}

	public void setFi1(ExpTree fi1) {
		m_fi1 = fi1;
	}

	public void setL(double l) {
		m_l = l;
	}

	public void setK(int k) {
		m_k = k;
	}

	public double getTau() {
		return m_tau;
	}

	public void setTau(double tau) {
		m_tau = tau;
	}

	public void setN(int n) {
		m_n = n;
	}

	public ExpTree getU() {
		m_u.setVar("a", m_a);
		m_u.setVar("b", m_b);
		m_u.setVar("c", m_c);

		return m_u;
	}

	public void setU(ExpTree u) {
		m_u = u;
	}

	public void solve(int schemeType, int boundCondType, Matrix matU, Vector vecX) {
		double h = m_l / m_n;
		//double t = m_tau * m_k;
		//double sigma = m_a * m_tau / (h * h);

		matU.resize(m_k + 1, m_n + 1);
		vecX.resize(m_n + 1);

		m_fi0.setVar("a", m_a);
		m_fi0.setVar("b", m_b);
		m_fi0.setVar("c", m_c);

		m_fi1.setVar("a", m_a);
		m_fi1.setVar("b", m_b);
		m_fi1.setVar("c", m_c);

		for (int j = 0; j <= m_n; ++j) {
			vecX.set(j, j * h);
			matU.set(0, j, m_psi.setVar("x", vecX.get(j)).calculate());
		}

		for (int i = 0; i < m_k; ++i) {
			for (int j = 1; j < m_n; ++j) {
				double res = 0.0;

				res += m_a * (matU.get(i, j + 1) - 2.0 * matU.get(i, j) + matU.get(i, j - 1)) / (h * h);
				res += m_b * (matU.get(i, j + 1) - matU.get(i, j - 1)) / (2.0 * h);
				res += m_c * matU.get(i, j);
				res += m_f.setVar("x", vecX.get(j)).setVar("t", i * m_tau).calculate();
				res *= m_tau;
				res += matU.get(i, j);

				matU.set(i + 1, j, res);
			}

			double fi0 = m_fi0.setVar("t", (i + 1) * m_tau).calculate();
			double fi1 = m_fi1.setVar("t", (i + 1) * m_tau).calculate();
			double bound1 = (h * fi0 - m_alpha * matU.get(i + 1, 1)) / (h * m_beta - m_alpha);
			double bound2 = (h * fi1 + m_gamma * matU.get(i + 1, m_n - 1)) / (h * m_delta + m_gamma);

			matU.set(i + 1, 0, bound1);
			matU.set(i + 1, m_n, bound2);
		}
	}

	private double m_a;
	private double m_b;
	private double m_c;
	private ExpTree m_f;
	private ExpTree m_psi;
	private double m_alpha;
	private double m_beta;
	private double m_gamma;
	private double m_delta;
	private ExpTree m_fi0;
	private ExpTree m_fi1;
	private double m_l;
	private int m_k;
	private double m_tau;
	private int m_n;
	private ExpTree m_u;

	public static final int SCHEME_EXPLICIT = 0;
	public static final int SCHEME_IMPLICIT = 1;
	public static final int SCHEME_CRANK_NICOLSON = 2;

	public static final int BOUNDARY_CONDITION_2_1 = 0;
	public static final int BOUNDARY_CONDITION_3_2 = 1;
	public static final int BOUNDARY_CONDITION_2_2 = 2;
}
