package libnm.math.pde;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class Elliptic {
	public void setBx(double bx) {
		m_bx = bx;
	}

	public void setBy(double by) {
		m_by = by;
	}

	public void setC(double c) {
		m_c = c;
	}

	public void setExprF(ExpTree exprF) {
		m_exprF = exprF;
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

	public void setNx(int nx) {
		m_nx = nx;
	}

	public void setNy(int ny) {
		m_ny = ny;
	}

	public void setOmega(double omega) {
		m_omega = omega;
	}

	public void setEps(double eps) {
		m_eps = eps;
	}

	public void setExprU(ExpTree exprU) {
		m_exprU = exprU;
	}

	public void solve(int methodType, Matrix matU, Vector vecX, Vector vecY) {
		int iterations = 0;
		double error = m_eps + 1.0;
		double hx = m_lx / m_nx;
		double hy = m_ly / m_ny;
		Matrix matTmp = new Matrix();

		matU.resize(m_ny + 1, m_nx + 1);
		matTmp.resize(m_ny + 1, m_nx + 1);
		vecX.resize(m_nx + 1);
		vecY.resize(m_ny + 1);

		for (int j = 0; j <= m_nx; ++j) {
			vecX.set(j, j * hx);
		}

		for (int i = 0; i <= m_ny; ++i) {
			vecY.set(i, i * hy);
		}

		while (error > m_eps) {
			error = 0.0;

			for (int i = 1; i < m_ny; ++i) {
				for (int j = 1; j < m_nx; ++j) {
					double res = 0.0;

					switch (methodType) {
						case METHOD_LIEBMANN: {
							res += (matU.get(i, j + 1) + matU.get(i, j - 1)) / (hx * hx);
							res += (matU.get(i + 1, j) + matU.get(i - 1, j)) / (hy * hy);
							res += m_bx * (matU.get(i, j + 1) - matU.get(i, j - 1)) / (2.0 * hx);
							res += m_by * (matU.get(i + 1, j) - matU.get(i - 1, j)) / (2.0 * hy);

							break;
						}

						case METHOD_SEIDEL: {
							res += (matU.get(i, j + 1) + matTmp.get(i, j - 1)) / (hx * hx);
							res += (matU.get(i + 1, j) + matTmp.get(i - 1, j)) / (hy * hy);
							res += m_bx * (matU.get(i, j + 1) - matTmp.get(i, j - 1)) / (2.0 * hx);
							res += m_by * (matU.get(i + 1, j) - matTmp.get(i - 1, j)) / (2.0 * hy);

							break;
						}
					}

					res += m_f(vecX.get(j), vecY.get(i));
					res /= 2.0 / (hx * hx) + 2.0 / (hy * hy) - m_c;
					res = (1 - m_omega) * matU.get(i, j) + res * m_omega; // SOR

					matTmp.set(i, j, res);
					matTmp.set(0, j, m_boundTopByX(matTmp, vecX, hy, j));
					matTmp.set(m_ny, j, m_boundDownByX(matTmp, vecX, hy, j));

					error = Math.max(error, Math.abs(matTmp.get(i, j) - matU.get(i, j)));
					error = Math.max(error, Math.abs(matTmp.get(0, j) - matU.get(0, j)));
					error = Math.max(error, Math.abs(matTmp.get(m_ny, j) - matU.get(m_ny, j)));
				}

				matTmp.set(i, 0, m_boundLeftByY(matTmp, vecY, hx, i));
				matTmp.set(i, m_nx, m_boundRightByY(matTmp, vecY, hx, i));

				error = Math.max(error, Math.abs(matTmp.get(i, 0) - matU.get(i, 0)));
				error = Math.max(error, Math.abs(matTmp.get(i, m_nx) - matU.get(i, m_nx)));
			}

			matU.copy(matTmp);
			++iterations;

			if (iterations > MAX_ITERATIONS) {
				break;
			}
		}

		matU.set(0, 0, m_boundLeftByY(matU, vecY, hx, 0));
		matU.set(0, m_nx, m_boundRightByY(matU, vecY, hx, 0));
		matU.set(m_ny, 0, m_boundLeftByY(matU, vecY, hx, m_ny));
		matU.set(m_ny, m_nx, m_boundRightByY(matU, vecY, hx, m_ny));
	}

	public double u(double x, double y) {
		m_exprU.setVar("bx", m_bx);
		m_exprU.setVar("by", m_by);
		m_exprU.setVar("c", m_c);
		m_exprU.setVar("x", x);
		m_exprU.setVar("y", y);

		return m_exprU.calculate();
	}

	private double m_f(double x, double y) {
		m_exprF.setVar("bx", m_bx);
		m_exprF.setVar("by", m_by);
		m_exprF.setVar("c", m_c);
		m_exprF.setVar("x", x);
		m_exprF.setVar("y", y);

		return m_exprF.calculate();
	}

	private double m_fi1(double y) {
		m_exprFi1.setVar("bx", m_bx);
		m_exprFi1.setVar("by", m_by);
		m_exprFi1.setVar("c", m_c);
		m_exprFi1.setVar("y", y);

		return m_exprFi1.calculate();
	}

	private double m_fi2(double y) {
		m_exprFi2.setVar("bx", m_bx);
		m_exprFi2.setVar("by", m_by);
		m_exprFi2.setVar("c", m_c);
		m_exprFi2.setVar("y", y);

		return m_exprFi2.calculate();
	}

	private double m_fi3(double x) {
		m_exprFi3.setVar("bx", m_bx);
		m_exprFi3.setVar("by", m_by);
		m_exprFi3.setVar("c", m_c);
		m_exprFi3.setVar("x", x);

		return m_exprFi3.calculate();
	}

	private double m_fi4(double x) {
		m_exprFi4.setVar("bx", m_bx);
		m_exprFi4.setVar("by", m_by);
		m_exprFi4.setVar("c", m_c);
		m_exprFi4.setVar("x", x);

		return m_exprFi4.calculate();
	}

	private double m_boundLeftByY(Matrix mat, Vector vecY, double hx, int indY) {
		return (m_fi1(vecY.get(indY)) - mat.get(indY, 1) * m_alpha1 / hx) / (m_beta1 - m_alpha1 / hx);
	}

	private double m_boundRightByY(Matrix mat, Vector vecY, double hx, int indY) {
		return (m_fi2(vecY.get(indY)) + mat.get(indY, m_nx - 1) * m_alpha2 / hx) / (m_beta2 + m_alpha2 / hx);
	}

	private double m_boundTopByX(Matrix mat, Vector vecX, double hy, int indX) {
		return (m_fi3(vecX.get(indX)) - mat.get(1, indX) * m_alpha3 / hy) / (m_beta3 - m_alpha3 / hy);
	}

	private double m_boundDownByX(Matrix mat, Vector vecX, double hy, int indX) {
		return (m_fi4(vecX.get(indX)) + mat.get(m_ny - 1, indX) * m_alpha4 / hy) / (m_beta4 + m_alpha4 / hy);
	}

	private double m_bx;
	private double m_by;
	private double m_c;
	private ExpTree m_exprF;
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
	private int m_nx;
	private int m_ny;
	private double m_omega;
	private double m_eps;
	private ExpTree m_exprU;

	public static final int METHOD_LIEBMANN = 0;
	public static final int METHOD_SEIDEL = 1;

	private static final int MAX_ITERATIONS = 10000;
}
