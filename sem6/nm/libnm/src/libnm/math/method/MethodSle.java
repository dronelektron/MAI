package libnm.math.method;

import libnm.math.*;
import libnm.util.Logger;

public class MethodSle {
	public MethodSle() {
		m_detSign = 1;
		m_logger = null;
	}

	public void setLogger(Logger logger) {
		m_logger = logger;
	}

	public boolean lup(Matrix mat, Vector vec, Vector vecX) {
		int n = mat.getSize();
		Matrix matL = new Matrix(n);
		Matrix matU = new Matrix(n);
		Vector vecP = new Vector(n);
		Vector vecY = new Vector(n);

		if (!m_lup(mat, matL, matU, vecP)) {
			return false;
		}

		if (m_logger != null) {
			m_logger.writeln("Матрица L:\n" + matL);
			m_logger.writeln("Матрица U:\n" + matU);
		}

		m_frontSub(matL, vec, vecP, vecY);
		m_backSub(matU, vecY, vecX);

		return true;
	}

	public boolean tma(Matrix mat, Vector vec, Vector vecX) {
		if (!m_tmaCheck(mat)) {
			return false;
		}

		int n = mat.getSize();
		Vector vecP = new Vector(n);
		Vector vecQ = new Vector(n);

		vecP.set(0, -mat.get(0, 1) / mat.get(0, 0));
		vecQ.set(0, vec.get(0) / mat.get(0, 0));

		for (int i = 1; i < n - 1; ++i) {
			double a = mat.get(i, i - 1);
			double b = mat.get(i, i);
			double c = mat.get(i, i + 1);

			vecP.set(i, -c / (b + a * vecP.get(i - 1)));
			vecQ.set(i, (vec.get(i) - a * vecQ.get(i - 1)) / (b + a * vecP.get(i - 1)));
		}

		double resUp = vec.get(n - 1) - mat.get(n - 1, n - 2) * vecQ.get(n - 2);
		double resDown = mat.get(n - 1, n - 1) + mat.get(n - 1, n - 2) * vecP.get(n - 2);

		vecQ.set(n - 1, resUp / resDown);
		vecX.set(n - 1, vecQ.get(n - 1));

		for (int i = n - 2; i >= 0; --i) {
			vecX.set(i, vecP.get(i) * vecX.get(i + 1) + vecQ.get(i));
		}

		if (m_logger != null) {
			m_logger.writeln("Вектор P:\n" + vecP);
			m_logger.writeln("Вектор Q:\n" + vecQ);
		}

		return true;
	}

	public void simpleIteration(Matrix mat, Vector vec, Vector vecX, double eps) {
		int n = mat.getSize();
		Matrix matA = new Matrix(n);
		Vector vecB = new Vector(n);
		Vector vecPrev = new Vector(n);
		int iterCnt = 1;

		vecB.copy(vec);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				if (i != j) {
					matA.set(i, j, -mat.get(i, j) / mat.get(i, i));
				}
			}

			vecB.set(i, vecB.get(i) / mat.get(i, i));
		}

		vecPrev.copy(vecB);

		if (m_logger != null) {
			m_logger.writeln("Матрица A:\n" + matA);
			m_logger.writeln("Вектор B:\n" + vecB);
		}

		while (true) {
			vecX.copy(matA.mul(vecPrev).add(vecB));

			if (m_logger != null) {
				m_logger.writeln("Итерация #" + iterCnt + ": " + vecX);
			}

			if (vecX.sub(vecPrev).normC() <= eps) {
				break;
			}

			vecPrev.copy(vecX);
			++iterCnt;

			if (iterCnt > MAX_ITERATIONS) {
				if (m_logger != null) {
					m_logger.writeln("Превышен лимит итераций");
				}

				break;
			}
		}
	}

	public void seidel(Matrix mat, Vector vec, Vector vecX, double eps) {
		int n = mat.getSize();
		Matrix matA = new Matrix(n);
		Matrix matB = new Matrix(n);
		Matrix matC = new Matrix(n);
		Vector vecB = new Vector(n);
		Vector vecPrev = new Vector(n);
		int iterCnt = 1;

		vecB.copy(vec);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				if (i != j) {
					matA.set(i, j, -mat.get(i, j) / mat.get(i, i));
				}
			}

			vecB.set(i, vecB.get(i) / mat.get(i, i));
		}

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				if (j < i) {
					matB.set(i, j, matA.get(i, j));
				} else {
					matC.set(i, j, matA.get(i, j));
				}
			}
		}

		vecPrev.copy(vecB);

		if (m_logger != null) {
			m_logger.writeln("Матрица A:\n" + matA);
			m_logger.writeln("Вектор B:\n" + vecB);
		}

		while (true) {
			vecX.copy(matB.mul(vecX).add(matC.mul(vecPrev)).add(vecB));

			if (m_logger != null) {
				m_logger.writeln("Итерация #" + iterCnt + ": " + vecX);
			}

			if (vecX.sub(vecPrev).normC() <= eps) {
				break;
			}

			vecPrev.copy(vecX);
			++iterCnt;

			if (iterCnt > MAX_ITERATIONS) {
				if (m_logger != null) {
					m_logger.writeln("Превышен лимит итераций");
				}

				break;
			}
		}
	}

	public boolean rotation(Matrix mat, Matrix matX, Vector vecX, double eps) {
		if (!m_rotationCheck(mat)) {
			return false;
		}

		int n = mat.getSize();
		Matrix matA = new Matrix(n);
		int iterCnt = 1;

		matA.copy(mat);
		matX.identity();

		while (true) {
			double max = 0.0;
			int maxRow = -1;
			int maxCol = -1;

			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					if (i < j) {
						double element = Math.abs(matA.get(i, j));

						if (element > max) {
							max = element;
							maxRow = i;
							maxCol = j;
						}
					}
				}
			}

			double a1 = matA.get(maxRow, maxRow);
			double a2 = matA.get(maxCol, maxCol);
			double fiZero = Math.PI / 4.0;

			if (a1 != a2) {
				fiZero = 0.5 * Math.atan(2.0 * matA.get(maxRow, maxCol) / (a1 - a2));
			}

			double fiCos = Math.cos(fiZero);
			double fiSin = Math.sin(fiZero);
			Matrix matU = new Matrix(n);

			matU.identity();
			matU.set(maxRow, maxRow, fiCos);
			matU.set(maxRow, maxCol, -fiSin);
			matU.set(maxCol, maxRow, fiSin);
			matU.set(maxCol, maxCol, fiCos);

			matX.copy(matX.mul(matU));
			matA = matU.transpose().mul(matA).mul(matU);

			if (m_logger != null) {
				m_logger.writeln("Итерация #" + iterCnt + ":\n" + matA);
			}

			double sum = 0.0;

			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					if (i < j) {
						sum += Math.pow(matA.get(i, j), 2.0);
					}
				}
			}

			if (Math.sqrt(sum) <= eps) {
				break;
			}

			++iterCnt;

			if (iterCnt > MAX_ITERATIONS) {
				if (m_logger != null) {
					m_logger.writeln("Превышен лимит итераций");
				}

				break;
			}
		}

		for (int i = 0; i < n; ++i) {
			vecX.set(i, matA.get(i, i));
		}

		return true;
	}

	public void qr(Matrix mat, Complex[] res, double eps) {
		int n = mat.getSize();
		Matrix matA = new Matrix(n);
		Complex[] prev = new Complex[n];
		boolean[] isComplex = new boolean[n];
		int iterCnt = 0;
		double error = eps + 1.0;

		for (int i = 0; i < n; ++i) {
			isComplex[i] = true;
		}

		for (int i = 0; i < n; ++i) {
			prev[i] = new Complex(1e+6, 0.0);
		}

		if ((n & 1) == 1) {
			isComplex[n - 1] = false;
		}

		matA.copy(mat);

		if (m_logger != null) {
			Matrix matQ = new Matrix(n);
			Matrix matR = new Matrix(n);

			m_qr(matA, matQ, matR);
			m_logger.writeln("Матрица Q * R:\n" + matQ.mul(matR));
			m_logger.writeln("Матрица Q * Q^T:\n" + matQ.mul(matQ.transpose()));
		}

		while (error > eps) {
			Matrix matQ = new Matrix(n);
			Matrix matR = new Matrix(n);
			double errorRat = 0.0;
			double errorCom = 0.0;

			++iterCnt;

			m_qr(matA, matQ, matR);
			matA = matR.mul(matQ);

			if (m_logger != null) {
				m_logger.writeln("Итерация #" + iterCnt);
				m_logger.writeln("Матрица A:\n" + matA);
			}

			for (int j = 0; j < n; ++j) {
				if (isComplex[j]) {
					if (m_qrNorm(matA, j) <= eps) {
						if (j + 2 < n) {
							for (int i = n - 1; i > j; --i) {
								isComplex[i] = isComplex[i - 1];
								prev[i] = prev[i - 1];
								res[i] = res[i - 1];
							}
						} else {
							isComplex[n - 1] = false;
							prev[n - 1] = null;
						}

						isComplex[j] = false;
						prev[j] = null;
						res[j] = new Complex(0.0, 0.0);
						--j;
					} else {
						m_qrSolveBlock(matA, j, res[j], res[j + 1]);

						errorCom = Math.max(errorCom, res[j].sub(prev[j]).abs());
						errorCom = Math.max(errorCom, res[j + 1].sub(prev[j + 1]).abs());

						++j;
					}
				} else {
					res[j].setRe(matA.get(j, j));
					errorRat = Math.max(errorRat, m_qrNorm(matA, j));
				}
			}

			if (m_logger != null) {
				for (int i = 0; i < n; ++i) {
					m_logger.writeln("Lambda #" + (i + 1) + ": " + res[i]);
				}

				m_logger.writeln("errorRat: " + errorRat);
				m_logger.writeln("errorCom: " + errorCom);
			}

			for (int i = 0; i < n; ++i) {
				if (isComplex[i]) {
					prev[i].setRe(res[i].getRe());
					prev[i].setIm(res[i].getIm());
				}
			}

			error = Math.max(errorRat, errorCom);

			if (iterCnt > MAX_ITERATIONS) {
				if (m_logger != null) {
					m_logger.writeln("Превышен лимит итераций");
				}

				break;
			}
		}
	}

	public void matInverse(Matrix mat, Matrix matInv) {
		int n = mat.getSize();
		Vector vec1 = new Vector(n);
		Vector vec2 = new Vector(n);
		Vector vecP = new Vector(n);
		Vector vecX = new Vector(n);
		Matrix matL = new Matrix(n);
		Matrix matU = new Matrix(n);
		Matrix matE = new Matrix(n);

		m_lup(mat, matL, matU, vecP);
		matE.identity();

		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < n; ++i) {
				vec1.set(i, matE.get(i, j));
			}

			m_frontSub(matL, vec1, vecP, vec2);
			m_backSub(matU, vec2, vecX);

			for (int i = 0; i < n; ++i) {
				matInv.set(i, j, vecX.get(i));
			}
		}

		if (m_logger != null) {
			m_logger.writeln("Матрица A * A^-1:\n" + mat.mul(matInv));
		}
	}

	public double matDet(Matrix mat) {
		int n = mat.getSize();
		double res = 1.0;
		Matrix matL = new Matrix(n);
		Matrix matU = new Matrix(n);
		Vector vecP = new Vector(n);

		m_lup(mat, matL, matU, vecP);

		for (int i = 0; i < n; ++i) {
			res *= matU.get(i, i);
		}

		return res * m_detSign;
	}

	private boolean m_lup(Matrix mat, Matrix matL, Matrix matU, Vector vecP) {
		int n = mat.getSize();

		m_detSign = 1;
		matU.copy(mat);

		for (int i = 0; i < n; ++i) {
			vecP.set(i, i);
		}

		for (int j = 0; j < n; ++j) {
			int row = -1;
			double max = 0.0;

			for (int i = j; i < n; ++i) {
				double element = Math.abs(matU.get(i, j));

				if (element > max) {
					max = element;
					row = i;
				}
			}

			if (row == -1) {
				return false;
			}

			if (row != j) {
				m_detSign *= -1;
			}

			matU.swapRows(j, row);
			matL.swapRows(j, row);
			matL.set(j, j, 1);
			vecP.swap(j, row);

			for (int i = j + 1; i < n; ++i) {
				double ratio = matU.get(i, j) / matU.get(j, j);

				for (int k = j; k < n; ++k) {
					matU.set(i, k, matU.get(i, k) - matU.get(j, k) * ratio);
				}

				matL.set(i, j, ratio);
			}
		}

		return true;
	}

	private void m_qr(Matrix mat, Matrix matQ, Matrix matR) {
		int n = mat.getSize();
		Matrix matA = new Matrix(n);
		Matrix matE = new Matrix(n);
		Matrix matResQ = new Matrix(n);

		matA.copy(mat);
		matE.identity();
		matResQ.identity();

		for (int j = 0; j < n - 1; ++j) {
			Vector vec = new Vector(n);
			Matrix ratioBottom = new Matrix(n);
			double sum = 0.0;

			for (int i = j; i < n; ++i) {
				sum += Math.pow(matA.get(i, j), 2.0);
			}

			vec.set(j, matA.get(j, j) + (matA.get(j, j) >= 0.0 ? 1.0 : -1.0) * Math.sqrt(sum));

			for (int i = j + 1; i < n; ++i) {
				vec.set(i, matA.get(i, j));
			}

			sum = 0.0;

			for (int i = 0; i < n; ++i) {
				sum += Math.pow(vec.get(i), 2.0);

				for (int k = 0; k < n; ++k) {
					ratioBottom.set(i, k, vec.get(i) * vec.get(k));
				}
			}

			Matrix matH = matE.sub(ratioBottom.mul(2.0 / sum));

			matA = matH.mul(matA);
			matResQ = matResQ.mul(matH);
		}

		matQ.copy(matResQ);
		matR.copy(matA);
	}

	private void m_qrSolveBlock(Matrix mat, int col, Complex c1, Complex c2) {
		double b = -(mat.get(col, col) + mat.get(col + 1, col + 1));
		double c = mat.get(col, col) * mat.get(col + 1, col + 1) - mat.get(col, col + 1) * mat.get(col + 1, col);
		double d = Math.pow(b, 2.0) - 4.0 * c;

		if (d >= 0.0) {
			double dRoot = Math.sqrt(d);

			c1.setRe((-b - dRoot) / 2.0);
			c2.setRe((-b + dRoot) / 2.0);
		} else {
			double dRoot = Math.sqrt(-d);

			c1.setRe(-b / 2.0);
			c1.setIm(-dRoot / 2.0);
			c2.setRe(-b / 2.0);
			c2.setIm(dRoot / 2.0);
		}
	}

	private double m_qrNorm(Matrix mat, int col) {
		double res = 0.0;

		for (int i = col + 1; i < mat.getSize(); ++i) {
			res += Math.pow(mat.get(i, col), 2.0);
		}

		return Math.sqrt(res);
	}

	private void m_frontSub(Matrix mat, Vector vec, Vector vecP, Vector vecX) {
		int n = mat.getSize();

		for (int i = 0; i < n; ++i) {
			double sum = 0.0;

			for (int j = 0; j < i; ++j) {
				sum += mat.get(i, j) * vecX.get(j);
			}

			vecX.set(i, vec.get((int)vecP.get(i)) - sum);
		}
	}

	private void m_backSub(Matrix mat, Vector vec, Vector vecX) {
		int n = mat.getSize();

		for (int i = n - 1; i >= 0; --i) {
			double sum = 0.0;

			for (int j = i + 1; j < n; ++j) {
				sum += mat.get(i, j) * vecX.get(j);
			}

			vecX.set(i, (vec.get(i) - sum) / mat.get(i, i));
		}
	}

	private boolean m_tmaCheck(Matrix mat) {
		int n = mat.getSize();

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				if (i == j - 1 || i == j || i == j + 1) {
					if (mat.get(i, j) == 0.0) {
						return false;
					}
				} else if (mat.get(i, j) != 0.0) {
					return false;
				}
			}
		}

		if (Math.abs(mat.get(0, 0)) < Math.abs(mat.get(0, 1)) ||
				Math.abs(mat.get(n - 1, n - 1)) < Math.abs(mat.get(n - 1, n - 2))) {
			return false;
		}

		for (int i = 1; i < n - 1; ++i) {
			if (Math.abs(mat.get(i, i)) < Math.abs(mat.get(i, i - 1)) + Math.abs(mat.get(i, i + 1))) {
				return false;
			}
		}

		return true;
	}

	private boolean m_rotationCheck(Matrix mat) {
		int n = mat.getSize();

		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				if (mat.get(i, j) != mat.get(j, i)) {
					return false;
				}
			}
		}

		return true;
	}

	private final int MAX_ITERATIONS = 1000;
	private int m_detSign;
	private Logger m_logger;
}
