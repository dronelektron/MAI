package math;

import java.awt.*;
import libnm.math.expression.ExpTree;
import libnm.math.method.MethodDerivate;
import libnm.math.method.MethodError;
import libnm.math.method.MethodIntegral;
import libnm.util.*;
import libnm.math.Vector;
import libnm.math.polynom.*;

public class Splines {
	public void method1() {
		m_method1("src/data/input/in1_a.txt", "src/data/output/out1_a.txt");
		m_method1("src/data/input/in1_b.txt", "src/data/output/out1_b.txt");
	}

	public void method2() {
		Reader reader = new Reader("src/data/input/in2.txt");
		Logger output = new Logger("src/data/output/out2.txt");
		Vector vecX = reader.readVector();
		Vector vecY = reader.readVector();
		double x = reader.readDouble();

		output.writeln("Метод 2: Кубический сплайн");
		output.writeln("X: " + vecX);
		output.writeln("Y: " + vecY);
		output.writeln("X*=" + x);

		PolynomCubic polyC = new PolynomCubic(vecX, vecY);

		output.writeln("Сплайн S(x):");
		output.writeln(polyC.toString());
		output.writeln("S(X*)=" + polyC.getValue(x));
		output.close();
		reader.close();
	}

	public void method3() {
		Reader reader = new Reader("src/data/input/in3.txt");
		Logger output = new Logger("src/data/output/out3.txt");
		Vector vecX = reader.readVector();
		Vector vecY = reader.readVector();
		Vector vecTmp = new Vector(vecX.getSize());
		Plotter plot = new Plotter(512.0, 512.0);

		plot.addData(vecX, vecY, Color.RED, "f(x)");

		output.writeln("Метод 3: МНК-аппроксимация");
		output.writeln("X: " + vecX);
		output.writeln("Y: " + vecY);

		PolynomMNK polyM = new PolynomMNK(vecX, vecY, 1);

		output.writeln("Полиномы:");
		output.writeln("P1(x)=" + polyM);
		output.writeln("e1=" + polyM.getSumOfSquares());

		for (int i = 0; i < vecTmp.getSize(); ++i) {
			vecTmp.set(i, polyM.getValue(vecX.get(i)));
		}

		plot.addData(vecX, vecTmp, Color.GREEN, "P1(x)");

		polyM = new PolynomMNK(vecX, vecY, 2);

		output.writeln("P2(x)=" + polyM);
		output.writeln("e2=" + polyM.getSumOfSquares());

		for (int i = 0; i < vecTmp.getSize(); ++i) {
			vecTmp.set(i, polyM.getValue(vecX.get(i)));
		}

		plot.addData(vecX, vecTmp, Color.BLUE, "P2(x)");
		plot.savePng("src/data/plot/plot3.png");

		output.close();
		reader.close();
	}

	public void method4() {
		Reader reader = new Reader("src/data/input/in4.txt");
		Logger output = new Logger("src/data/output/out4.txt");
		Vector vecX = reader.readVector();
		Vector vecY = reader.readVector();
		double x = reader.readDouble();
		MethodDerivate method = new MethodDerivate(vecX, vecY, x);

		output.writeln("Метод 4: Численное дифференцирование");
		output.writeln("X: " + vecX);
		output.writeln("Y: " + vecY);
		output.writeln("X*=" + x);
		output.writeln("f'(X*)=" + method.deriv1());
		output.writeln("f''(X*)=" + method.deriv2());
		output.close();
		reader.close();
	}

	public void method5() {
		Reader reader = new Reader("src/data/input/in5.txt");
		Logger output = new Logger("src/data/output/out5.txt");
		String line = reader.readLine();
		ExpTree expr = new ExpTree(line);
		double x0 = reader.readDouble();
		double x1 = reader.readDouble();
		double h1 = reader.readDouble();
		double h2 = reader.readDouble();
		MethodIntegral method = new MethodIntegral(expr, x0, x1, h1);

		output.writeln("Метод 5: Численное интегрирование");
		output.writeln("Функция:" + line);
		output.writeln("x0=" + x0);
		output.writeln("x1=" + x1);

		double rect1 = method.rectangle();

		output.writeln("h1=" + h1);
		output.writeln("Прямоугольник: " + rect1);

		method.setH(h2);

		double rect2 = method.rectangle();
		double rectError = rect1 + MethodError.rungeRomberg(h1, h2, rect1, rect2, 2.0);

		output.writeln("h2=" + h2);
		output.writeln("Прямоугольник: " + rect2);
		output.writeln("Уточнение методом Рунге-Ромберга: " + rectError);

		method.setH(h1);

		double trap1 = method.trapezoidal();

		output.writeln("h1=" + h1);
		output.writeln("Трапеция: " + trap1);

		method.setH(h2);

		double trap2 = method.trapezoidal();
		double trapError = trap1 + MethodError.rungeRomberg(h1, h2, trap1, trap2, 2.0);

		output.writeln("h2=" + h2);
		output.writeln("Трапеция: " + trap2);
		output.writeln("Уточнение методом Рунге-Ромберга: " + trapError);

		method.setH(h1);

		double simp1 = method.simpson();

		output.writeln("h1=" + h1);
		output.writeln("Симпсон: " + simp1);

		method.setH(h2);

		double simp2 = method.simpson();
		double simpError = simp1 + MethodError.rungeRomberg(h1, h2, simp1, simp2, 2.0);

		output.writeln("h2=" + h2);
		output.writeln("Симпсон: " + simp2);
		output.writeln("Уточнение методом Рунге-Ромберга: " + simpError);

		output.close();
		reader.close();
	}

	private void m_method1(String inputName, String outputName) {
		Reader reader = new Reader(inputName);
		Logger output = new Logger(outputName);
		Vector vecX = reader.readVector();
		String func = reader.readLine();
		double x = reader.readDouble();
		ExpTree expr = new ExpTree(func);

		output.writeln("Метод 1: Интерполяция");
		output.writeln("X: " + vecX);
		output.writeln("X*=" + x);

		PolynomLagrange polyL = new PolynomLagrange(vecX, expr);
		PolynomNewton polyN = new PolynomNewton(vecX, expr);

		output.writeln("Полиномы:");
		output.writeln("L(x)=" + polyL);
		output.writeln("e(X)=" + Math.abs(polyL.getValue(x) - expr.setVar("x", x).calculate()));
		output.writeln("y(X)=" + polyL.getValue(x));
		output.writeln("N(x)=" + polyN);
		output.writeln("e(X)=" + Math.abs(polyN.getValue(x) - expr.setVar("x", x).calculate()));
		output.writeln("y(X)=" + polyN.getValue(x));
		output.close();
		reader.close();
	}
}
