package libnm.math;

import java.util.Arrays;

public class Vector {
	public Vector() {
		this(1);
	}

	public Vector(int size) {
		m_vec = new double[size];
	}

	public Vector(double[] array) {
		m_vec = array.clone();
	}

	public void resize(int size) {
		m_vec = new double[size];
	}

	public double get(int index) {
		return m_vec[index];
	}

	public void set(int index, double value) {
		m_vec[index] = value;
	}

	public int getSize() {
		return m_vec.length;
	}

	public Vector add(Vector other) {
		Vector res = new Vector(getSize());

		for (int i = 0; i < getSize(); ++i) {
			res.set(i, get(i) + other.get(i));
		}

		return res;
	}

	public Vector sub(Vector other) {
		Vector res = new Vector(getSize());

		for (int i = 0; i < getSize(); ++i) {
			res.set(i, get(i) - other.get(i));
		}

		return res;
	}

	public double normC() {
		double res = 0.0;

		for (int i = 0; i < getSize(); ++i) {
			double curX = Math.abs(get(i));

			if (curX > res) {
				res = curX;
			}
		}

		return res;
	}

	public void copy(Vector other) {
		for (int i = 0; i < getSize(); ++i) {
			set(i, other.get(i));
		}
	}

	public void swap(int index1, int index2) {
		double tmp = get(index1);

		set(index1, get(index2));
		set(index2, tmp);
	}

	@Override
	public String toString() {
		return Arrays.toString(m_vec);
	}

	private double[] m_vec;
}
