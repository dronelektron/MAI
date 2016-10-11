package libnm.math;

import java.util.Arrays;

public class Matrix {
	public Matrix() {
		this(1);
	}

	public Matrix(int size) {
		m_mat = new double[size][size];
	}

	public Matrix(int rows, int cols) {
		resize(rows, cols);
	}

	public void resize(int rows, int cols) {
		m_mat = new double[rows][cols];
	}

	public double get(int row, int col) {
		return m_mat[row][col];
	}

	public void set(int row, int col, double value) {
		m_mat[row][col] = value;
	}

	public int getSize() {
		return m_mat.length;
	}

	public int getM() {
		return getSize();
	}

	public int getN() {
		return m_mat[0].length;
	}

	public Matrix sub(Matrix other) {
		int n = getSize();
		Matrix res = new Matrix(n);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				res.set(i, j, get(i, j) - other.get(i, j));
			}
		}

		return res;
	}

	public Matrix mul(double value) {
		int n = getSize();
		Matrix res = new Matrix(n);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				res.set(i, j, get(i, j) * value);
			}
		}

		return res;
	}

	public Matrix mul(Matrix other) {
		int n = getSize();
		Matrix res = new Matrix(n);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				double sum = 0.0;

				for (int k = 0; k < n; ++k) {
					sum += get(i, k) * other.get(k, j);
				}

				res.set(i, j, sum);
			}
		}

		return res;
	}

	public Vector mul(Vector other) {
		int n = getSize();
		Vector res = new Vector(n);

		for (int i = 0; i < n; ++i) {
			double sum = 0.0;

			for (int j = 0; j < n; ++j) {
				sum += get(i, j) * other.get(j);
			}

			res.set(i, sum);
		}

		return res;
	}

	public Matrix transpose() {
		int n = getSize();
		Matrix res = new Matrix(n);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				res.set(i, j, get(j, i));
			}
		}

		return res;
	}

	public void identity() {
		int n = getSize();

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				if (i == j) {
					set(i, j, 1.0);
				} else {
					set(i, j, 0.0);
				}
			}
		}
	}

	public void copy(Matrix other) {
		for (int i = 0; i < getM(); ++i) {
			for (int j = 0; j < getN(); ++j) {
				set(i, j, other.get(i, j));
			}
		}
	}

	public void swapRows(int index1, int index2) {
		double[] tmp = m_mat[index1];

		m_mat[index1] = m_mat[index2];
		m_mat[index2] = tmp;
	}

	@Override
	public String toString() {
		String res = Arrays.toString(m_mat[0]);

		for (int i = 1; i < getM(); ++i) {
			res += '\n' + Arrays.toString(m_mat[i]);
		}

		return res;
	}

	private double[][] m_mat;
}
