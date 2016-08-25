package main;

import java.awt.*;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;
import libnm.math.pde.Hyperbolic;
import libnm.util.*;

public class Lab6 {
	public Lab6() {
		Reader reader = new Reader("src/data/input/in1.txt");

		m_method = new Hyperbolic();
		m_method.setA(reader.readDouble());
		m_method.setB(reader.readDouble());
		m_method.setC(reader.readDouble());
		m_method.setE(reader.readDouble());
		m_method.setExprF(new ExpTree(reader.readLine()));
		m_method.setExprPsi1(new ExpTree(reader.readLine()));
		m_method.setExprPsi2(new ExpTree(reader.readLine()));
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

	public void method111() {
		m_methodTemplate("111", Hyperbolic.SCHEME_EXPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_1, Hyperbolic.INITIAL_CONDITION_1);
	}

	public void method112() {
		m_methodTemplate("112", Hyperbolic.SCHEME_EXPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_1, Hyperbolic.INITIAL_CONDITION_2);
	}

	public void method121() {
		m_methodTemplate("121", Hyperbolic.SCHEME_EXPLICIT, Hyperbolic.BOUNDARY_CONDITION_3_2, Hyperbolic.INITIAL_CONDITION_1);
	}

	public void method122() {
		m_methodTemplate("122", Hyperbolic.SCHEME_EXPLICIT, Hyperbolic.BOUNDARY_CONDITION_3_2, Hyperbolic.INITIAL_CONDITION_2);
	}

	public void method131() {
		m_methodTemplate("131", Hyperbolic.SCHEME_EXPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_2, Hyperbolic.INITIAL_CONDITION_1);
	}

	public void method132() {
		m_methodTemplate("132", Hyperbolic.SCHEME_EXPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_2, Hyperbolic.INITIAL_CONDITION_2);
	}

	public void method211() {
		m_methodTemplate("211", Hyperbolic.SCHEME_IMPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_1, Hyperbolic.INITIAL_CONDITION_1);
	}

	public void method212() {
		m_methodTemplate("212", Hyperbolic.SCHEME_IMPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_1, Hyperbolic.INITIAL_CONDITION_2);
	}

	public void method221() {
		m_methodTemplate("221", Hyperbolic.SCHEME_IMPLICIT, Hyperbolic.BOUNDARY_CONDITION_3_2, Hyperbolic.INITIAL_CONDITION_1);
	}

	public void method222() {
		m_methodTemplate("222", Hyperbolic.SCHEME_IMPLICIT, Hyperbolic.BOUNDARY_CONDITION_3_2, Hyperbolic.INITIAL_CONDITION_2);
	}

	public void method231() {
		m_methodTemplate("231", Hyperbolic.SCHEME_IMPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_2, Hyperbolic.INITIAL_CONDITION_1);
	}

	public void method232() {
		m_methodTemplate("232", Hyperbolic.SCHEME_IMPLICIT, Hyperbolic.BOUNDARY_CONDITION_2_2, Hyperbolic.INITIAL_CONDITION_2);
	}

	private void m_methodTemplate(String postFixName, int schemeType, int boundCondType, int initCondType) {
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

		m_method.solve(schemeType, boundCondType, initCondType, matRes, vecX, vecT);

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

	private Hyperbolic m_method;
}
