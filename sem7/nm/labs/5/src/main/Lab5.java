package main;

import java.awt.*;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;
import libnm.math.pde.Parabolic;
import libnm.util.*;

public class Lab5 {
	public Lab5() {
		Reader reader = new Reader("src/data/input/in10.txt");

		m_method = new Parabolic();
		m_method.setA(reader.readDouble());
		m_method.setB(reader.readDouble());
		m_method.setC(reader.readDouble());
		m_method.setExprF(new ExpTree(reader.readLine()));
		m_method.setExprPsi(new ExpTree(reader.readLine()));
		m_method.setAlpha(reader.readDouble());
		m_method.setBeta(reader.readDouble());
		m_method.setGamma(reader.readDouble());
		m_method.setDelta(reader.readDouble());
		m_method.setExprFi0(new ExpTree(reader.readLine()));
		m_method.setExprFi1(new ExpTree(reader.readLine()));
		m_method.setL(reader.readDouble());
		m_method.setK(reader.readInt());
		m_method.setTau(reader.readDouble());
		m_method.setN(reader.readInt());
		m_method.setExprU(new ExpTree(reader.readLine()));

		reader.close();
	}

	public void method11() {
		m_methodTemplate("11", Parabolic.SCHEME_EXPLICIT, Parabolic.BOUNDARY_CONDITION_2_1);
	}

	public void method12() {
		m_methodTemplate("12", Parabolic.SCHEME_EXPLICIT, Parabolic.BOUNDARY_CONDITION_3_2);
	}

	public void method13() {
		m_methodTemplate("13", Parabolic.SCHEME_EXPLICIT, Parabolic.BOUNDARY_CONDITION_2_2);
	}

	public void method21() {
		m_methodTemplate("21", Parabolic.SCHEME_IMPLICIT, Parabolic.BOUNDARY_CONDITION_2_1);
	}

	public void method22() {
		m_methodTemplate("22", Parabolic.SCHEME_IMPLICIT, Parabolic.BOUNDARY_CONDITION_3_2);
	}

	public void method23() {
		m_methodTemplate("23", Parabolic.SCHEME_IMPLICIT, Parabolic.BOUNDARY_CONDITION_2_2);
	}

	public void method31() {
		m_methodTemplate("31", Parabolic.SCHEME_CRANK_NICOLSON, Parabolic.BOUNDARY_CONDITION_2_1);
	}

	public void method32() {
		m_methodTemplate("32", Parabolic.SCHEME_CRANK_NICOLSON, Parabolic.BOUNDARY_CONDITION_3_2);
	}

	public void method33() {
		m_methodTemplate("33", Parabolic.SCHEME_CRANK_NICOLSON, Parabolic.BOUNDARY_CONDITION_2_2);
	}

	private void m_methodTemplate(String postFixName, int schemeType, int boundCondType) {
		Plotter plotter = new Plotter(512.0, 512.0);
		Plotter plotterErr = new Plotter(512.0, 512.0);
		Logger output = new Logger("src/data/output/out" + postFixName + ".txt");
		Matrix matRes = new Matrix();
		Vector vecX = new Vector();
		Vector vecT = new Vector();
		Vector vecY1 = new Vector();
		Vector vecY2 = new Vector();
		Vector vecErr = new Vector();
		int k = 100;

		m_method.solve(schemeType, boundCondType, matRes, vecX, vecT);

		output.writeln(matRes.toString());
		output.close();

		vecY1.resize(vecX.getSize());
		vecY2.resize(vecX.getSize());
		vecErr.resize(matRes.getM());

		for (int j = 0; j < vecX.getSize(); ++j) {
			vecY1.set(j, m_method.u(vecX.get(j), vecT.get(k)));
			vecY2.set(j, matRes.get(k, j));
		}

		plotter.addData(vecX, vecY1, Color.RED, "U(x, t) (анал.)");
		plotter.addData(vecX, vecY2, Color.BLUE, "U(x, t) (числ.)");
		plotter.savePng("src/data/plot/plot" + postFixName + ".png");

		for (int i = 0; i < matRes.getM(); ++i) {
			double error = 0.0;

			for (int j = 0; j < matRes.getN(); ++j) {
				error += Math.abs(matRes.get(i, j) - m_method.u(vecX.get(j), vecT.get(i)));
			}

			vecErr.set(i, error);
		}

		plotterErr.addData(vecT, vecErr, Color.RED, "e(x)");
		plotterErr.savePng("src/data/plot/plot" + postFixName + "_error.png");
	}

	private Parabolic m_method;
}
