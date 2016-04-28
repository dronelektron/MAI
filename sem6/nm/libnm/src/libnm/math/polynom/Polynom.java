package libnm.math.polynom;

import libnm.math.Vector;

public class Polynom extends Vector {
	public Polynom(Vector vecX) {
		super(vecX.getSize());

		for (int i = 0; i < vecX.getSize(); ++i) {
			set(i, vecX.get(i));
		}
	}

	public double getValue(double x) {
		double res = 0.0;

		for (int i = 0; i < getSize(); ++i) {
			res += get(i) * Math.pow(x, i);
		}

		return res;
	}

	@Override
	public String toString() {
		String res = String.valueOf(get(0));

		for (int i = 1; i < getSize(); ++i) {
			if (get(i) >= 0.0) {
				res += "+";
			}

			res += get(i) + "x^" + i;
		}

		return res;
	}
}
