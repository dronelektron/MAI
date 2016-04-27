package libnm.math.method;

import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class MethodError {
	public static double calcError(ExpTree expr, Vector vecX, Vector vecY) {
		double res = 0.0;

		for (int i = 0; i < vecX.getSize(); ++i) {
			expr.setVar("x", vecX.get(i));
			res += Math.abs(expr.calculate() - vecY.get(i));
		}

		return res;
	}

	public static double rungeRomberg(double h1, double h2, double y1, double y2, double p) {
		double r = h2 / h1;

		return (y1 - y2) / (Math.pow(r, p) - 1.0);
	}
}
