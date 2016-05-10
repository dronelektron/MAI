package libnm.math.polynom;

import libnm.math.Matrix;
import libnm.math.method.MethodSle;
import libnm.math.Vector;

public class PolynomMNK {
	public PolynomMNK(Vector vecX, Vector vecY, int degree) {
		int n = vecX.getSize();
		int m = degree + 1;
		int sumCnt = degree * 2 + 1;
		double[] sumsMat = new double[sumCnt];
		double[] sumsVec = new double[m];
		Matrix mat = new Matrix(m);
		Vector vec = new Vector(m);
		Vector vecA = new Vector(m);
		MethodSle method = new MethodSle();

		m_vecX = vecX;
		m_vecY = vecY;

		for (int i = 0; i < sumCnt; ++i) {
			for (int j = 0; j < n; ++j) {
				sumsMat[i] += Math.pow(m_vecX.get(j), i);
			}
		}

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				sumsVec[i] += m_vecY.get(j) * Math.pow(m_vecX.get(j), i);
			}
		}

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				mat.set(i, j, sumsMat[i + j]);
			}
		}

		for (int i = 0; i < m; ++i) {
			vec.set(i, sumsVec[i]);
		}

		method.lup(mat, vec, vecA);
		m_poly = new Polynom(vecA);
	}

	public double getSumOfSquares() {
		double res = 0.0;

		for (int i = 0; i < m_vecX.getSize(); ++i) {
			res += Math.pow(m_poly.getValue(m_vecX.get(i)) - m_vecY.get(i), 2.0);
		}

		return res;
	}

	public double getValue(double x) {
		return m_poly.getValue(x);
	}

	@Override
	public String toString() {
		return m_poly.toString();
	}

	private Vector m_vecX;
	private Vector m_vecY;
	private Polynom m_poly;
}
