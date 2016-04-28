package libnm.math.polynom;

import libnm.math.Matrix;
import libnm.math.method.MethodSle;
import libnm.math.Vector;

public class PolynomMNK extends Polynom {
	public PolynomMNK(Vector vecX, Vector vecY, int degree) {
		super(vecX);

		int n = getSize();
		int m = degree + 1;
		int sumCnt = degree * 2 + 1;
		double[] sumsMat = new double[sumCnt];
		double[] sumsVec = new double[m];
		Matrix mat = new Matrix(m);
		Vector vec = new Vector(m);
		Vector vecA = new Vector(m);
		MethodSle method = new MethodSle();

		m_vec = new Vector(getSize());
		m_vec.copy(vecY);

		for (int i = 0; i < sumCnt; ++i) {
			for (int j = 0; j < n; ++j) {
				sumsMat[i] += Math.pow(get(j), i);
			}
		}

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				sumsVec[i] += m_vec.get(j) * Math.pow(get(j), i);
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

		for (int i = 0; i < getSize(); ++i) {
			res += Math.pow(m_poly.getValue(get(i)) - m_vec.get(i), 2.0);
		}

		return res;
	}

	@Override
	public double getValue(double x) {
		return m_poly.getValue(x);
	}

	@Override
	public String toString() {
		return m_poly.toString();
	}

	private Vector m_vec;
	private Polynom m_poly;
}
