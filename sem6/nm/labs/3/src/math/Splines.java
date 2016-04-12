package math;

import libnm.math.expression.ExpTree;
import libnm.math.method.MethodDerivate;
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

		output.writeln("Метод 3: МНК-аппроксимация");
		output.writeln("X: " + vecX);
		output.writeln("Y: " + vecY);

		PolynomMNK polyM = new PolynomMNK(vecX, vecY, 1);

		output.writeln("Полиномы:");
		output.writeln("P1(x)=" + polyM);
		output.writeln("e1=" + polyM.getSumOfSquares());

		polyM = new PolynomMNK(vecX, vecY, 2);

		output.writeln("P2(x)=" + polyM);
		output.writeln("e2=" + polyM.getSumOfSquares());
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
		output.writeln("h1=" + h1);
		output.writeln("Прямоугольник: " + method.rectangle());
		output.writeln("Трапеция: " + method.trapezoidal());
		output.writeln("Симпсон: " + method.simpson());

		method.setH(h2);

		output.writeln("h2=" + h2);
		output.writeln("Прямоугольник: " + method.rectangle());
		output.writeln("Трапеция: " + method.trapezoidal());
		output.writeln("Симпсон: " + method.simpson());

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
		PolynomNewton polyN = new PolynomNewton(vecX);

		output.writeln("Полиномы:");
		output.writeln("L(x)=" + polyL);
		output.writeln("e(X)=" + Math.abs(polyL.getValue(x) - expr.setVar("x", x).calculate()));
		output.writeln("N(x)=" + polyN);
		output.writeln("e(X)=" + Math.abs(polyN.getValue(x) - polyN.func(x)));
		output.close();
		reader.close();
	}
}
