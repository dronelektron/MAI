package libnm.math.pde;

import java.util.ArrayList;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;
import libnm.math.method.MethodSle;

public class Parabolic2D {
	public void setA(double a) {
		m_a = a;
	}

	public void setB(double b) {
		m_b = b;
	}

	public void setExprF(ExpTree exprF) {
		m_exprF = exprF;
	}

	public void setExprPsi(ExpTree exprPsi) {
		m_exprPsi = exprPsi;
	}

	public void setAlpha1(double alpha1) {
		m_alpha1 = alpha1;
	}

	public void setBeta1(double beta1) {
		m_beta1 = beta1;
	}

	public void setAlpha2(double alpha2) {
		m_alpha2 = alpha2;
	}

	public void setBeta2(double beta2) {
		m_beta2 = beta2;
	}

	public void setAlpha3(double alpha3) {
		m_alpha3 = alpha3;
	}

	public void setBeta3(double beta3) {
		m_beta3 = beta3;
	}

	public void setAlpha4(double alpha4) {
		m_alpha4 = alpha4;
	}

	public void setBeta4(double beta4) {
		m_beta4 = beta4;
	}

	public void setExprFi1(ExpTree exprFi1) {
		m_exprFi1 = exprFi1;
	}

	public void setExprFi2(ExpTree exprFi2) {
		m_exprFi2 = exprFi2;
	}

	public void setExprFi3(ExpTree exprFi3) {
		m_exprFi3 = exprFi3;
	}

	public void setExprFi4(ExpTree exprFi4) {
		m_exprFi4 = exprFi4;
	}

	public void setLx(double lx) {
		m_lx = lx;
	}

	public void setLy(double ly) {
		m_ly = ly;
	}

	public void setTau(double tau) {
		m_tau = tau;
	}

	public void setNx(int nx) {
		m_nx = nx;
	}

	public void setNy(int ny) {
		m_ny = ny;
	}

	public void setNt(int nt) {
		m_nt = nt;
	}

	public void setExprU(ExpTree exprU) {
		m_exprU = exprU;
	}

	public void solve(int methodType, ArrayList<Matrix> matU, Vector vecX, Vector vecY, Vector vecT) {
		double tau2 = m_tau / 2.0;
		double hx = m_lx / m_nx;
		double hy = m_ly / m_ny;

		vecX.resize(m_nx + 1);
		vecY.resize(m_ny + 1);
		vecT.resize(m_nt + 1);
		matU.add(new Matrix(m_ny + 1, m_nx + 1));

		for (int j = 0; j <= m_nx; ++j) {
			vecX.set(j, j * hx);
		}

		for (int i = 0; i <= m_ny; ++i) {
			vecY.set(i, i * hy);
		}

		for (int t = 0; t <= m_nt; ++t) {
			vecT.set(t, t * m_tau);
		}

		for (int i = 0; i <= m_ny; ++i) {
			for (int j = 0; j <= m_nx; ++j) {
				matU.get(0).set(i, j, m_psi(vecX.get(j), vecY.get(i)));
			}
		}

		switch (methodType) {
			case METHOD_ALTERNATING_DIRECTION: {
				double coefAx = tau2 * m_a / (hx * hx);
				double coefBx = -2.0 * coefAx - 1.0;
				double coefCx = coefAx;
				double coefAy = tau2 * m_b / (hy * hy);
				double coefBy = -2.0 * coefAy - 1.0;
				double coefCy = coefAy;
				MethodSle sleSolver = new MethodSle();

				for (int k = 0; k < m_nt; ++k) {
					matU.add(new Matrix(m_ny + 1, m_nx + 1));

					double tHalf = vecT.get(k) + tau2;
					double tNext = vecT.get(k + 1);
					Matrix matTmp = new Matrix(m_ny + 1, m_nx + 1);
					Matrix matCur = matU.get(k);
					Matrix matNext = matU.get(k + 1);

					for (int i = 1; i < m_ny; ++i) {
						Matrix mat = new Matrix(m_nx + 1);
						Vector vec = new Vector(m_nx + 1);
						Vector vecRes = new Vector(m_nx + 1);

						for (int j = 1; j < m_nx; ++j) {
							double res = 0.0;

							res -= matCur.get(i, j);
							res -= (tau2 * m_b / (hy * hy)) * (matCur.get(i + 1, j) - 2.0 * matCur.get(i, j) + matCur.get(i + 1, j));
							res -= tau2 * m_f(vecX.get(j), vecY.get(i), tHalf);

							mat.set(j, j - 1, coefAx);
							mat.set(j, j, coefBx);
							mat.set(j, j + 1, coefCx);
							vec.set(j, res);
						}

						mat.set(0, 0, m_beta1 - m_alpha1 / hx);
						mat.set(0, 1, m_alpha1 / hx);
						mat.set(m_nx, m_nx - 1, -m_alpha2 / hx);
						mat.set(m_nx, m_nx, m_beta2 + m_alpha2 / hx);

						vec.set(0, m_fi1(vecY.get(i), tHalf));
						vec.set(m_nx, m_fi2(vecY.get(i), tHalf));

						sleSolver.tma(mat, vec, vecRes, false);

						for (int j = 0; j <= m_nx; ++j) {
							matTmp.set(i, j, vecRes.get(j));
						}
					}

					matCur = matTmp;

					for (int j = 1; j < m_nx; ++j) {
						Matrix mat = new Matrix(m_ny + 1);
						Vector vec = new Vector(m_ny + 1);
						Vector vecRes = new Vector(m_ny + 1);

						for (int i = 1; i < m_ny; ++i) {
							double res = 0.0;

							res -= matCur.get(i, j);
							res -= (tau2 * m_a / (hx * hx)) * (matCur.get(i, j + 1) - 2.0 * matCur.get(i, j) + matCur.get(i, j - 1));
							res -= tau2 * m_f(vecX.get(j), vecY.get(i), tHalf);

							mat.set(i, i - 1, coefAx);
							mat.set(i, i, coefBx);
							mat.set(i, i + 1, coefCx);
							vec.set(i, res);
						}

						mat.set(0, 0, m_beta3 - m_alpha3 / hy);
						mat.set(0, 1, m_alpha3 / hy);
						mat.set(m_ny, m_ny - 1, -m_alpha4 / hy);
						mat.set(m_ny, m_ny, m_beta4 + m_alpha4 / hx);

						vec.set(0, m_fi3(vecX.get(j), tHalf));
						vec.set(m_ny, m_fi4(vecX.get(j), tHalf));

						sleSolver.tma(mat, vec, vecRes, false);

						for (int i = 0; i <= m_ny; ++i) {
							matNext.set(i, j, vecRes.get(i));
						}
					}

					for (int i = 0; i <= m_ny; ++i) {
						matNext.set(i, 0, m_leftBound(vecY.get(i), tNext, hx, matCur.get(i, 1)));
						matNext.set(i, m_nx, m_rightBound(vecY.get(i), tNext, hx, matCur.get(i, m_nx - 1)));
					}
				}

				break;
			}

			case METHOD_FRACTIONAL_STEP: {


				break;
			}
		}
	}

	public double u(double x, double y, double t) {
		m_exprU.setVar("a", m_a);
		m_exprU.setVar("b", m_b);
		m_exprU.setVar("x", x);
		m_exprU.setVar("y", y);
		m_exprU.setVar("t", t);

		return m_exprU.calculate();
	}

	private double m_f(double x, double y, double t) {
		m_exprF.setVar("a", m_a);
		m_exprF.setVar("b", m_b);
		m_exprF.setVar("x", x);
		m_exprF.setVar("y", y);
		m_exprF.setVar("t", t);

		return m_exprF.calculate();
	}

	private double m_psi(double x, double y) {
		m_exprPsi.setVar("a", m_a);
		m_exprPsi.setVar("b", m_b);
		m_exprPsi.setVar("x", x);
		m_exprPsi.setVar("y", y);

		return m_exprPsi.calculate();
	}

	private double m_fi1(double y, double t) {
		m_exprFi1.setVar("a", m_a);
		m_exprFi1.setVar("b", m_b);
		m_exprFi1.setVar("y", y);
		m_exprFi1.setVar("t", t);

		return m_exprFi1.calculate();
	}

	private double m_fi2(double y, double t) {
		m_exprFi2.setVar("a", m_a);
		m_exprFi2.setVar("b", m_b);
		m_exprFi2.setVar("y", y);
		m_exprFi2.setVar("t", t);

		return m_exprFi2.calculate();
	}

	private double m_fi3(double x, double t) {
		m_exprFi3.setVar("a", m_a);
		m_exprFi3.setVar("b", m_b);
		m_exprFi3.setVar("x", x);
		m_exprFi3.setVar("t", t);

		return m_exprFi3.calculate();
	}

	private double m_fi4(double x, double t) {
		m_exprFi4.setVar("a", m_a);
		m_exprFi4.setVar("b", m_b);
		m_exprFi4.setVar("x", x);
		m_exprFi4.setVar("t", t);

		return m_exprFi4.calculate();
	}

	private double m_leftBound(double y, double t, double hx, double arg) {
		return (m_fi1(y, t) - (m_alpha1 / hx) * arg) / (m_beta1 - m_alpha1 / hx);
	}

	private double m_rightBound(double y, double t, double hx, double arg) {
		return (m_fi2(y, t) + (m_alpha2 / hx) * arg) / (m_beta2 + m_alpha2 / hx);
	}

	private double m_a;
	private double m_b;
	private ExpTree m_exprF;
	private ExpTree m_exprPsi;
	private double m_alpha1;
	private double m_beta1;
	private double m_alpha2;
	private double m_beta2;
	private double m_alpha3;
	private double m_beta3;
	private double m_alpha4;
	private double m_beta4;
	private ExpTree m_exprFi1;
	private ExpTree m_exprFi2;
	private ExpTree m_exprFi3;
	private ExpTree m_exprFi4;
	private double m_lx;
	private double m_ly;
	private double m_tau;
	private int m_nx;
	private int m_ny;
	private int m_nt;
	private ExpTree m_exprU;

	public static final int METHOD_ALTERNATING_DIRECTION = 0;
	public static final int METHOD_FRACTIONAL_STEP = 1;
}
