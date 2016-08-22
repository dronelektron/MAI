package libnm.math.pde;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;
import libnm.math.method.MethodSle;

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

	public void setExprF(ExpTree exprF) {
		m_exprF = exprF;
	}

	public void setExprPsi(ExpTree exprPsi) {
		m_exprPsi = exprPsi;
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

	public void setExprFi0(ExpTree exprFi0) {
		m_exprFi0 = exprFi0;
	}

	public void setExprFi1(ExpTree exprFi1) {
		m_exprFi1 = exprFi1;
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

	public void setExprU(ExpTree exprU) {
		m_exprU = exprU;
	}

	public void solve(int schemeType, int boundCondType, Matrix matU, Vector vecX) {
		double h = m_l / m_n;

		matU.resize(m_k + 1, m_n + 1);
		vecX.resize(m_n + 1);

		for (int j = 0; j <= m_n; ++j) {
			vecX.set(j, j * h);
			matU.set(0, j, m_psi(vecX.get(j)));
		}

		for (int i = 0; i < m_k; ++i) {
			switch (schemeType) {
				case SCHEME_EXPLICIT: {
					Vector vecRes = new Vector(m_n + 1);

					m_explicit(matU, vecX, i, h, boundCondType, vecRes);
					m_copyVectorToMatrix(vecRes, matU, i + 1);

					break;
				}

				case SCHEME_IMPLICIT: {
					Vector vecRes = new Vector(m_n + 1);

					m_implicit(matU, vecX, i, h, boundCondType, vecRes);
					m_copyVectorToMatrix(vecRes, matU, i + 1);

					break;
				}

				case SCHEME_CRANK_NICOLSON: {
					double theta = 0.5;
					Vector vecResExp = new Vector(m_n + 1);
					Vector vecResImp = new Vector(m_n + 1);

					m_explicit(matU, vecX, i, h, boundCondType, vecResExp);
					m_implicit(matU, vecX, i, h, boundCondType, vecResImp);

					for (int j = 0; j <= m_n; ++j) {
						matU.set(i + 1, j, theta * vecResImp.get(j) + (1.0 - theta) * vecResExp.get(j));
					}

					break;
				}
			}
		}
	}

	public double u(double x, double t) {
		m_exprU.setVar("a", m_a);
		m_exprU.setVar("b", m_b);
		m_exprU.setVar("c", m_c);
		m_exprU.setVar("x", x);
		m_exprU.setVar("t", t);

		return m_exprU.calculate();
	}

	private double m_f(double x, double t) {
		m_exprF.setVar("a", m_a);
		m_exprF.setVar("b", m_b);
		m_exprF.setVar("c", m_c);
		m_exprF.setVar("x", x);
		m_exprF.setVar("t", t);

		return m_exprF.calculate();
	}

	private double m_psi(double x) {
		m_exprPsi.setVar("a", m_a);
		m_exprPsi.setVar("b", m_b);
		m_exprPsi.setVar("c", m_c);
		m_exprPsi.setVar("x", x);

		return m_exprPsi.calculate();
	}

	private double m_fi0(double t) {
		m_exprFi0.setVar("a", m_a);
		m_exprFi0.setVar("b", m_b);
		m_exprFi0.setVar("c", m_c);
		m_exprFi0.setVar("t", t);

		return m_exprFi0.calculate();
	}

	private double m_fi1(double t) {
		m_exprFi1.setVar("a", m_a);
		m_exprFi1.setVar("b", m_b);
		m_exprFi1.setVar("c", m_c);
		m_exprFi1.setVar("t", t);

		return m_exprFi1.calculate();
	}

	private void m_explicit(Matrix matU, Vector vecX, int i, double h, int boundCondType, Vector vecRes) {
		for (int j = 1; j < m_n; ++j) {
			double res = 0.0;

			res += m_a * (matU.get(i, j + 1) - 2.0 * matU.get(i, j) + matU.get(i, j - 1)) / (h * h);
			res += m_b * (matU.get(i, j + 1) - matU.get(i, j - 1)) / (2.0 * h);
			res += m_c * matU.get(i, j);
			res += m_f(vecX.get(j), i * m_tau);
			res *= m_tau;
			res += matU.get(i, j);

			vecRes.set(j, res);
		}

		double bound1 = 0.0;
		double bound2 = 0.0;
		double tNext = (i + 1) * m_tau;
		double fi0 = m_fi0(tNext);
		double fi1 = m_fi1(tNext);

		switch (boundCondType) {
			case BOUNDARY_CONDITION_2_1:
				bound1 = (h * fi0 - m_alpha * vecRes.get(1)) / (h * m_beta - m_alpha);
				bound2 = (h * fi1 + m_gamma * vecRes.get(m_n - 1)) / (h * m_delta + m_gamma);

				break;

			case BOUNDARY_CONDITION_3_2:
				bound1 = 2.0 * h * fi0 - m_alpha * (4.0 * vecRes.get(1) - vecRes.get(2));
				bound1 /= 2.0 * h * m_beta - 3.0 * m_alpha;
				bound2 = 2.0 * h * fi1 - m_gamma * (vecRes.get(m_n - 2) - 4.0 * vecRes.get(m_n - 1));
				bound2 /= 2.0 * h * m_delta + 3.0 * m_gamma;

				break;

			case BOUNDARY_CONDITION_2_2:
				double b0 = 2.0 * m_a / h + h / m_tau - m_c * h - (m_beta / m_alpha) * (2.0 * m_a - m_b * h);
				double c0 = -2.0 * m_a / h;
				double d0 = (h / m_tau) * matU.get(i, 0) - m_fi0(tNext) * (2.0 * m_a - m_b * h) / m_alpha;
				double an = -2.0 * m_a / h;
				double bn = 2.0 * m_a / h + h / m_tau - m_c * h + (m_delta / m_gamma) * (2.0 * m_a + m_b * h);
				double dn = (h / m_tau) * matU.get(i, m_n) + m_fi1(tNext) * (2.0 * m_a + m_b * h) / m_gamma;

				bound1 = (d0 - c0 * vecRes.get(1)) / b0;
				bound2 = (dn - an * vecRes.get(m_n - 1)) / bn;

				break;
		}

		vecRes.set(0, bound1);
		vecRes.set(m_n, bound2);
	}

	private void m_implicit(Matrix matU, Vector vecX, int i, double h, int boundCondType, Vector vecRes) {
		MethodSle sleSolver = new MethodSle();
		Matrix mat = new Matrix(m_n + 1);
		Vector vec = new Vector(m_n + 1);
		double sigma1 = m_tau * m_a / (h * h);
		double sigma2 = m_tau * m_c;
		double sigma3 = m_tau * m_b / (2.0 * h);
		double coefA = sigma1 - sigma3;
		double coefB = sigma2 - 2.0 * sigma1 - 1.0;
		double coefC = sigma1 + sigma3;
		double tNext = (i + 1) * m_tau;

		switch (boundCondType) {
			case BOUNDARY_CONDITION_2_1:
				mat.set(0, 0, m_beta - m_alpha / h);
				mat.set(0, 1, m_alpha / h);
				mat.set(m_n, m_n - 1, -m_gamma / h);
				mat.set(m_n, m_n, m_delta + m_gamma / h);
				vec.set(0, m_fi0(tNext));
				vec.set(m_n, m_fi1(tNext));

				break;

			case BOUNDARY_CONDITION_3_2:
				double h2 = 2.0 * h;

				mat.set(0, 0, m_beta - 3.0 * m_alpha / h2);
				mat.set(0, 1, 4.0 * m_alpha / h2);
				mat.set(0, 2, -m_alpha / h2);
				mat.set(m_n, m_n - 2, m_gamma / h2);
				mat.set(m_n, m_n - 1, -4.0 * m_gamma / h2);
				mat.set(m_n, m_n, m_delta + 3.0 * m_gamma / h2);
				vec.set(0, m_fi0(tNext));
				vec.set(m_n, m_fi1(tNext));

				break;

			case BOUNDARY_CONDITION_2_2:
				if (m_alpha == 0.0) {
					mat.set(0, 0, m_beta);
					vec.set(0, m_fi0(tNext));
				} else {
					double b0 = 2.0 * m_a / h + h / m_tau - m_c * h - (m_beta / m_alpha) * (2.0 * m_a - m_b * h);
					double c0 = -2.0 * m_a / h;
					double d0 = (h / m_tau) * matU.get(i, 0) - m_fi0(tNext) * (2.0 * m_a - m_b * h) / m_alpha;

					mat.set(0, 0, b0);
					mat.set(0, 1, c0);
					vec.set(0, d0);
				}

				if (m_gamma == 0.0) {
					mat.set(m_n, m_n, m_delta);
					vec.set(m_n, m_fi1(tNext));
				} else {
					double an = -2.0 * m_a / h;
					double bn = 2.0 * m_a / h + h / m_tau - m_c * h + (m_delta / m_gamma) * (2.0 * m_a + m_b * h);
					double dn = (h / m_tau) * matU.get(i, m_n) + m_fi1(tNext) * (2.0 * m_a + m_b * h) / m_gamma;

					mat.set(m_n, m_n - 1, an);
					mat.set(m_n, m_n, bn);
					vec.set(m_n, dn);
				}

				break;
		}

		for (int row = 1; row < m_n; ++row) {
			mat.set(row, row - 1, coefA);
			mat.set(row, row, coefB);
			mat.set(row, row + 1, coefC);
			vec.set(row, -matU.get(i, row) - m_f(vecX.get(row), i * m_tau) * m_tau);
		}

		sleSolver.lup(mat, vec, vecRes);
	}

	private void m_copyVectorToMatrix(Vector vec, Matrix mat, int row) {
		for (int j = 0; j < vec.getSize(); ++j) {
			mat.set(row, j, vec.get(j));
		}
	}

	private double m_a;
	private double m_b;
	private double m_c;
	private ExpTree m_exprF;
	private ExpTree m_exprPsi;
	private double m_alpha;
	private double m_beta;
	private double m_gamma;
	private double m_delta;
	private ExpTree m_exprFi0;
	private ExpTree m_exprFi1;
	private double m_l;
	private int m_k;
	private double m_tau;
	private int m_n;
	private ExpTree m_exprU;

	public static final int SCHEME_EXPLICIT = 0;
	public static final int SCHEME_IMPLICIT = 1;
	public static final int SCHEME_CRANK_NICOLSON = 2;

	public static final int BOUNDARY_CONDITION_2_1 = 0;
	public static final int BOUNDARY_CONDITION_3_2 = 1;
	public static final int BOUNDARY_CONDITION_2_2 = 2;
}
