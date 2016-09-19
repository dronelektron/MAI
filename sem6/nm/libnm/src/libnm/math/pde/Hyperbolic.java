package libnm.math.pde;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;
import libnm.math.method.MethodSle;

public class Hyperbolic {
	public void setA(double a) {
		m_a = a;
	}

	public void setB(double b) {
		m_b = b;
	}

	public void setC(double c) {
		m_c = c;
	}

	public void setE(double e) {
		m_e = e;
	}

	public void setExprF(ExpTree exprF) {
		m_exprF = exprF;
	}

	public void setExprPsi1(ExpTree exprPsi1) {
		m_exprPsi1 = exprPsi1;
		m_exprPsi1Deriv1 = m_exprPsi1.derivateBy("x");
		m_exprPsi1Deriv2 = m_exprPsi1Deriv1.derivateBy("x");
	}

	public void setExprPsi2(ExpTree exprPsi2) {
		m_exprPsi2 = exprPsi2;
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

	public void setTau(double tau) {
		m_tau = tau;
	}

	public void setN(int n) {
		m_n = n;
	}

	public void setExprU(ExpTree exprU) {
		m_exprU = exprU;
	}

	public void solve(int schemeType, int boundCondType, int initCondType, Matrix matU, Vector vecX, Vector vecT) {
		double h = m_l / m_n;

		matU.resize(m_k + 1, m_n + 1);
		vecX.resize(m_n + 1);
		vecT.resize(m_k + 1);

		for (int j = 0; j <= m_n; ++j) {
			vecX.set(j, j * h);
			matU.set(0, j, m_psi1(vecX.get(j)));
		}

		for (int i = 0; i <= m_k; ++i) {
			vecT.set(i, i * m_tau);
		}

		switch (initCondType) {
			case INITIAL_CONDITION_1:
				for (int j = 0; j <= m_n; ++j) {
					matU.set(1, j, matU.get(0, j) + m_psi2(vecX.get(j)) * m_tau);
				}

				break;

			case INITIAL_CONDITION_2:
				for (int j = 0; j <= m_n; ++j) {
					double x = vecX.get(j);
					double res = 0.0;

					// TODO: Уточнить: vecT.get(0) или vecT.get(1)?
					res += m_a * m_a * m_psi1Deriv2(x) + m_b * m_psi1Deriv1(x) + m_c * matU.get(0, j) + m_f(x, vecT.get(0));
					res *= m_tau * m_tau / 2.0;
					res += m_psi2(x) * m_tau;
					res += matU.get(0, j);

					matU.set(1, j, res);
				}

				break;
		}

		for (int i = 1; i < m_k; ++i) {
			double tCur = vecT.get(i);
			double tNext = vecT.get(i + 1);

			switch (schemeType) {
				case SCHEME_EXPLICIT: {
					double fi0 = m_fi0(tCur);
					double fi1 = m_fi1(tCur);

					for (int j = 1; j < m_n; ++j) {
						double res = 0.0;

						res += m_a * m_a * (matU.get(i, j + 1) - 2.0 * matU.get(i, j) + matU.get(i, j - 1)) / (h * h);
						res += m_b * (matU.get(i, j + 1) - matU.get(i, j - 1)) / (2.0 * h);
						res += m_c * matU.get(i, j) + m_f(vecX.get(j), vecT.get(i));
						res *= 2.0 * m_tau * m_tau;
						res -= 2.0 * (matU.get(i - 1, j) - 2.0 * matU.get(i, j));
						res += m_e * m_tau * matU.get(i - 1, j);
						res /= 2.0 + m_e * m_tau;

						matU.set(i + 1, j, res);
					}

					double bound1 = 0.0;
					double bound2 = 0.0;

					switch (boundCondType) {
						case BOUNDARY_CONDITION_2_1:
							bound1 = (h * fi0 - m_alpha * matU.get(i + 1, 1)) / (h * m_beta - m_alpha);
							bound2 = (h * fi1 + m_gamma * matU.get(i + 1, m_n - 1)) / (h * m_delta + m_gamma);

							break;

						case BOUNDARY_CONDITION_3_2:
							bound1 = 2.0 * h * fi0 - m_alpha * (4.0 * matU.get(i + 1, 1) - matU.get(i + 1, 2));
							bound1 /= 2.0 * h * m_beta - 3.0 * m_alpha;
							bound2 = 2.0 * h * fi1 - m_gamma * (matU.get(i + 1, m_n - 2) - 4.0 * matU.get(i + 1, m_n - 1));
							bound2 /= 2.0 * h * m_delta + 3.0 * m_gamma;

							break;

						case BOUNDARY_CONDITION_2_2:
							if (m_alpha == 0.0) {
								bound1 = fi0 / m_beta;
							} else {
								double b0 = 2.0 * m_a * m_a / h + h / m_tau - m_c * h - (m_beta / m_alpha) * (2.0 * m_a * m_a - m_b * h);
								double c0 = -2.0 * m_a * m_a / h;
								double d0 = (h / m_tau) * matU.get(i, 0) - fi0 * (2.0 * m_a * m_a - m_b * h) / m_alpha;

								bound1 = (d0 - c0 * matU.get(i + 1, 1)) / b0;
							}

							if (m_gamma == 0.0) {
								bound2 = fi1 / m_delta;
							} else {
								double an = -2.0 * m_a * m_a / h;
								double bn = 2.0 * m_a * m_a / h + h / m_tau - m_c * h + (m_delta / m_gamma) * (2.0 * m_a * m_a + m_b * h);
								double dn = (h / m_tau) * matU.get(i, m_n) + fi1 * (2.0 * m_a * m_a + m_b * h) / m_gamma;

								bound2 = (dn - an * matU.get(i + 1, m_n - 1)) / bn;
							}

							break;
					}

					matU.set(i + 1, 0, bound1);
					matU.set(i + 1, m_n, bound2);

					break;
				}

				case SCHEME_IMPLICIT: {
					MethodSle sleSolver = new MethodSle();
					Matrix mat = new Matrix(m_n + 1);
					Vector vec = new Vector(m_n + 1);
					Vector vecRes = new Vector(m_n + 1);
					double sigma1 = m_tau * m_tau * m_a * m_a / (h * h);
					double sigma2 = m_tau * m_tau * m_c - 1.0 - m_e * m_tau / 2.0;
					double sigma3 = m_tau * m_tau * m_b / (2.0 * h);
					double coefA = sigma1 - sigma3;
					double coefB = sigma2 - 2.0 * sigma1;
					double coefC = sigma1 + sigma3;
					double fi0 = m_fi0(tNext);
					double fi1 = m_fi1(tNext);

					switch (boundCondType) {
						case BOUNDARY_CONDITION_2_1:
							mat.set(0, 0, m_beta - m_alpha / h);
							mat.set(0, 1, m_alpha / h);
							mat.set(m_n, m_n - 1, -m_gamma / h);
							mat.set(m_n, m_n, m_delta + m_gamma / h);
							vec.set(0, fi0);
							vec.set(m_n, fi1);

							break;

						case BOUNDARY_CONDITION_3_2:
							double h2 = 2.0 * h;

							mat.set(0, 0, m_beta - 3.0 * m_alpha / h2);
							mat.set(0, 1, 4.0 * m_alpha / h2);
							mat.set(0, 2, -m_alpha / h2);
							mat.set(m_n, m_n - 2, m_gamma / h2);
							mat.set(m_n, m_n - 1, -4.0 * m_gamma / h2);
							mat.set(m_n, m_n, m_delta + 3.0 * m_gamma / h2);
							vec.set(0, fi0);
							vec.set(m_n, fi1);

							break;

						case BOUNDARY_CONDITION_2_2:
							if (m_alpha == 0.0) {
								mat.set(0, 0, m_beta);
								vec.set(0, fi0);
							} else {
								double b0 = 2.0 * m_a * m_a / h + h / m_tau - m_c * h - (m_beta / m_alpha) * (2.0 * m_a * m_a - m_b * h);
								double c0 = -2.0 * m_a * m_a / h;
								double d0 = (h / m_tau) * matU.get(i, 0) - fi0 * (2.0 * m_a * m_a - m_b * h) / m_alpha;

								mat.set(0, 0, b0);
								mat.set(0, 1, c0);
								vec.set(0, d0);
							}

							if (m_gamma == 0.0) {
								mat.set(m_n, m_n, m_delta);
								vec.set(m_n, fi1);
							} else {
								double an = -2.0 * m_a * m_a / h;
								double bn = 2.0 * m_a * m_a / h + h / m_tau - m_c * h + (m_delta / m_gamma) * (2.0 * m_a * m_a + m_b * h);
								double dn = (h / m_tau) * matU.get(i, m_n) + fi1 * (2.0 * m_a * m_a + m_b * h) / m_gamma;

								mat.set(m_n, m_n - 1, an);
								mat.set(m_n, m_n, bn);
								vec.set(m_n, dn);
							}

							break;
					}

					for (int row = 1; row < m_n; ++row) {
						double res = 0.0;

						res += (1.0 - m_e * m_tau / 2.0) * matU.get(i - 1, row);
						res -= 2.0 * matU.get(i, row);
						res -= m_tau * m_tau * m_f(vecX.get(row), tNext);

						mat.set(row, row - 1, coefA);
						mat.set(row, row, coefB);
						mat.set(row, row + 1, coefC);
						vec.set(row, res);
					}

					sleSolver.lup(mat, vec, vecRes);

					for (int j = 0; j <= m_n; ++j) {
						matU.set(i + 1, j, vecRes.get(j));
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
		m_exprU.setVar("e1", m_e);
		m_exprU.setVar("x", x);
		m_exprU.setVar("t", t);

		return m_exprU.calculate();
	}

	private double m_f(double x, double t) {
		m_exprF.setVar("a", m_a);
		m_exprF.setVar("b", m_b);
		m_exprF.setVar("c", m_c);
		m_exprF.setVar("e1", m_e);
		m_exprF.setVar("x", x);
		m_exprF.setVar("t", t);

		return m_exprF.calculate();
	}

	private double m_psi1(double x) {
		m_exprPsi1.setVar("a", m_a);
		m_exprPsi1.setVar("b", m_b);
		m_exprPsi1.setVar("c", m_c);
		m_exprPsi1.setVar("e1", m_e);
		m_exprPsi1.setVar("x", x);

		return m_exprPsi1.calculate();
	}

	private double m_psi2(double x) {
		m_exprPsi2.setVar("a", m_a);
		m_exprPsi2.setVar("b", m_b);
		m_exprPsi2.setVar("c", m_c);
		m_exprPsi2.setVar("e1", m_e);
		m_exprPsi2.setVar("x", x);

		return m_exprPsi2.calculate();
	}

	private double m_psi1Deriv1(double x) {
		m_exprPsi1Deriv1.setVar("a", m_a);
		m_exprPsi1Deriv1.setVar("b", m_b);
		m_exprPsi1Deriv1.setVar("c", m_c);
		m_exprPsi1Deriv1.setVar("e1", m_e);
		m_exprPsi1Deriv1.setVar("x", x);

		return m_exprPsi1Deriv1.calculate();
	}

	private double m_psi1Deriv2(double x) {
		m_exprPsi1Deriv2.setVar("a", m_a);
		m_exprPsi1Deriv2.setVar("b", m_b);
		m_exprPsi1Deriv2.setVar("c", m_c);
		m_exprPsi1Deriv2.setVar("e1", m_e);
		m_exprPsi1Deriv2.setVar("x", x);

		return m_exprPsi1Deriv2.calculate();
	}

	private double m_fi0(double t) {
		m_exprFi0.setVar("a", m_a);
		m_exprFi0.setVar("b", m_b);
		m_exprFi0.setVar("c", m_c);
		m_exprFi0.setVar("e1", m_e);
		m_exprFi0.setVar("t", t);

		return m_exprFi0.calculate();
	}

	private double m_fi1(double t) {
		m_exprFi1.setVar("a", m_a);
		m_exprFi1.setVar("b", m_b);
		m_exprFi1.setVar("c", m_c);
		m_exprFi1.setVar("e1", m_e);
		m_exprFi1.setVar("t", t);

		return m_exprFi1.calculate();
	}

	private double m_a;
	private double m_b;
	private double m_c;
	private double m_e;
	private ExpTree m_exprF;
	private ExpTree m_exprPsi1;
	private ExpTree m_exprPsi2;
	private ExpTree m_exprPsi1Deriv1;
	private ExpTree m_exprPsi1Deriv2;
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

	public static final int BOUNDARY_CONDITION_2_1 = 0;
	public static final int BOUNDARY_CONDITION_3_2 = 1;
	public static final int BOUNDARY_CONDITION_2_2 = 2;

	public static final int INITIAL_CONDITION_1 = 0;
	public static final int INITIAL_CONDITION_2 = 1;
}
