package libnm.math.method;

import libnm.math.expression.ExpTree;
import libnm.math.*;
import libnm.util.Logger;

public class MethodSnle {
	public MethodSnle() {
		m_logger = null;
	}

	public void setLogger(Logger logger) {
		m_logger = logger;
	}

	public void simpleIteration(ExpTree[] exprs, Vector vecPrev, Vector vecX, double eps, double q) {
		int n = exprs.length;
		int iterCnt = 1;

		while (true) {
			Vector vecTmp = new Vector(n);

			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					exprs[i].setVar("x" + (j + 1), vecPrev.get(j));
				}

				vecTmp.set(i, exprs[i].calculate());
			}

			vecX.copy(vecTmp);

			if (m_logger != null) {
				m_logger.writeln("Итерация #" + iterCnt + ": " + vecX);
			}

			if (vecX.sub(vecPrev).normC() * q / (1.0 - q) <= eps) {
				break;
			}

			vecPrev.copy(vecX);
			++iterCnt;
		}
	}

	public void newton(ExpTree[] exprs, Vector vecPrev, Vector vecX, double eps) {
		int n = exprs.length;
		int iterCnt = 1;
		ExpTree[][] matJ = new ExpTree[n][n];
		MethodSle method = new MethodSle();

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				matJ[i][j] = exprs[i].derivateBy("x" + (j + 1));
			}
		}

		while (true) {
			Matrix mat = new Matrix(n);
			Vector vec = new Vector(n);
			Vector vecDelta = new Vector(n);

			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					for (int k = 0; k < n; ++k) {
						matJ[i][j].setVar("x" + (k + 1), vecPrev.get(k));
					}

					mat.set(i, j, matJ[i][j].calculate());
					exprs[i].setVar("x" + (j + 1), vecPrev.get(j));
				}

				vec.set(i, -exprs[i].calculate());
			}

			method.lup(mat, vec, vecDelta);
			vecX.copy(vecPrev.add(vecDelta));

			if (m_logger != null) {
				m_logger.writeln("Итерация #" + iterCnt + ": " + vecX);
			}

			if (vecX.sub(vecPrev).normC() <= eps) {
				break;
			}

			vecPrev.copy(vecX);
			++iterCnt;
		}
	}

	private Logger m_logger;
}
