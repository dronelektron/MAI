package libnm.math.method;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;

public class MethodDiffEqBoundary {
	public MethodDiffEqBoundary(ExpTree exprR, ExpTree exprP, ExpTree exprQ, ExpTree exprF, double a, double b, double h, double alpha, double beta, double delta, double gamma, double y0, double y1) {
		m_exprR = exprR;
		m_exprP = exprP;
		m_exprQ = exprQ;
		m_exprF = exprF;
		m_a = a;
		m_b = b;
		m_h = h;
		m_alpha = alpha;
		m_beta = beta;
		m_delta = delta;
		m_gamma = gamma;
		m_y0 = y0;
		m_y1 = y1;
	}

	public void shooting(Vector vecX, Vector vecY, double eps) {
		int n = getN();
		double yEta1;
		double yEta2;
		double eta1 = 1000000.0;
		double eta2 = -1000000.0;
		Vector vecZ = new Vector(n);
		String strExprR = m_exprR.getExpr();
		ExpTree exprP = new ExpTree(m_exprP.getExpr() + "/" + strExprR);
		ExpTree exprQ = new ExpTree(m_exprQ.getExpr() + "/" + strExprR);
		ExpTree exprF = new ExpTree(m_exprF.getExpr() + "/" + strExprR);

		yEta1 = fi(eta1, vecX, vecY, vecZ, exprP, exprQ, exprF);
		yEta2 = fi(eta2, vecX, vecY, vecZ, exprP, exprQ, exprF);

		double error = Math.abs(eta2 - eta1);

		while (error > eps) {
			double etaN = eta2 - yEta2 * (eta2 - eta1) / (yEta2 - yEta1);

			yEta1 = yEta2;
			yEta2 = fi(etaN, vecX, vecY, vecZ, exprP, exprQ, exprF);
			eta1 = eta2;
			eta2 = etaN;
			error = Math.abs(eta2 - eta1);
		}
	}

	public void finiteDifference(Vector vecX, Vector vecY) {
		int n = getN();
		Matrix mat = new Matrix(n);
		Vector vec = new Vector(n);
		MethodSle method = new MethodSle();

		vecX.set(0, m_a);

		for (int i = 1; i < n; ++i) {
			vecX.set(i, vecX.get(i - 1) + m_h);
		}

		//mat.set(0, 0, m_alpha - m_beta / m_h);
		//mat.set(0, 1, m_beta / m_h);
		mat.set(0, 0, m_alpha - 3.0 * m_beta / (2.0 * m_h));
		mat.set(0, 1, 4.0 * m_beta / (2.0 * m_h));
		mat.set(0, 2, -m_beta / (2.0 * m_h));
		vec.set(0, m_y0);

		for (int i = 1; i < n - 1; ++i) {
			double x = vecX.get(i);
			double rx = m_exprR.setVar("x", x).calculate();
			double px = m_exprP.setVar("x", x).calculate() / rx;
			double qx = m_exprQ.setVar("x", x).calculate() / rx;
			double fx = m_exprF.setVar("x", x).calculate() / rx;
			double a = 1.0 / (m_h * m_h) - px / (2.0 * m_h);
			double b = -2.0 / (m_h * m_h) + qx;
			double c = 1.0 / (m_h * m_h) + px / (2.0 * m_h);

			mat.set(i, i - 1, a);
			mat.set(i, i, b);
			mat.set(i, i + 1, c);
			vec.set(i, -fx);
		}

		//mat.set(n - 1, n - 2, -m_gamma / m_h);
		//mat.set(n - 1, n - 1, m_delta + m_gamma / m_h);
		mat.set(n - 1, n - 3, m_gamma / (2.0 * m_h));
		mat.set(n - 1, n - 2, -4.0 * m_gamma / (2.0 * m_h));
		mat.set(n - 1, n - 1, m_delta + 3.0 * m_gamma / (2.0 * m_h));
		vec.set(n - 1, m_y1);

		//method.tma(mat, vec, vecY);
		method.lup(mat, vec, vecY);
	}

	public int getN() {
		return (int)((m_b - m_a) / m_h) + 1;
	}

	public void setH(double h) {
		m_h = h;
	}

	private double fi(double eta, Vector vecX, Vector vecY, Vector vecZ, ExpTree exprP, ExpTree exprQ, ExpTree exprF) {
		int n = getN();
		double y0;
		double z0;
		MethodDiffEqCauchy method;

		if (m_beta != 0.0) {
			y0 = eta;
			z0 = (m_y0 - m_alpha * eta) / m_beta;
		} else {
			y0 = m_y0 / m_alpha;
			z0 = eta;
		}

		method = new MethodDiffEqCauchy(exprP, exprQ, exprF, y0, z0, m_a, m_b, m_h);
		method.rungeKutta(vecX, vecY, vecZ);

		return m_delta * vecY.get(n - 1) + m_gamma * vecZ.get(n - 1) - m_y1;
	}

	private ExpTree m_exprR;
	private ExpTree m_exprP;
	private ExpTree m_exprQ;
	private ExpTree m_exprF;
	private double m_a;
	private double m_b;
	private double m_alpha;
	private double m_beta;
	private double m_delta;
	private double m_gamma;
	private double m_y0;
	private double m_y1;
	private double m_h;
}
