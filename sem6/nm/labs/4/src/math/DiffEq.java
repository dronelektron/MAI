package math;

import java.awt.*;

import libnm.math.Vector;
import libnm.math.expression.ExpTree;
import libnm.math.method.MethodDiffEqBoundary;
import libnm.math.method.MethodDiffEqCauchy;
import libnm.math.method.MethodError;
import libnm.util.*;

public class DiffEq {
	public void method1() {
		Reader reader = new Reader("src/data/input/in1.txt");
		Logger output = new Logger("src/data/output/out1.txt");
		ExpTree exprP = new ExpTree(reader.readLine());
		ExpTree exprQ = new ExpTree(reader.readLine());
		ExpTree exprF = new ExpTree(reader.readLine());
		ExpTree exprA = new ExpTree(reader.readLine());
		double y0 = reader.readDouble();
		double z0 = reader.readDouble();
		double a = reader.readDouble();
		double b = reader.readDouble();
		double h = reader.readDouble();
		double h2 = h / 2.0;
		MethodDiffEqCauchy method = new MethodDiffEqCauchy(exprP, exprQ, exprF, y0, z0, a, b, h);
		int n = method.getN();
		Vector vecX1 = new Vector(n);
		Vector vecY1 = new Vector(n);
		Vector vecZ1 = new Vector(n);
		Vector vecX2 = new Vector(n * 2);
		Vector vecY2 = new Vector(n * 2);
		Vector vecZ2 = new Vector(n * 2);
		Vector vecY = new Vector(n);

		method.euler(vecX1, vecY1);
		method.setH(h2);
		method.euler(vecX2, vecY2);
		method.setH(h);

		for (int i = 0; i < n; ++i) {
			vecY.set(i, exprA.setVar("x", vecX1.get(i)).calculate());
		}

		output.writeln("Метод 1: Метод Эйлера");
		output.writeln("Решение:");
		output.writeln("X: " + vecX1);
		output.writeln("Y: " + vecY1);
		output.writeln("Точное решение:");
		output.writeln("Y: " + vecY);
		output.writeln("Погрешность относительно точного решения:\n" + MethodError.calcError(exprA, vecX1, vecY1));
		output.writeln("Погрешность методом Рунге-Ромберга:");

		for (int i = 0; i < n; ++i) {
			output.writeln("X[" + i + "]: " + MethodError.rungeRomberg(h, h2, vecY1.get(i), vecY2.get(i * 2), 2.0));
		}

		method.rungeKutta(vecX1, vecY1, vecZ1);
		method.setH(h2);
		method.rungeKutta(vecX2, vecY2, vecZ2);
		method.setH(h);

		output.writeln("Метод 2: Метод Рунге-Кутты");
		output.writeln("Решение:");
		output.writeln("X: " + vecX1);
		output.writeln("Y: " + vecY1);
		output.writeln("Точное решение:");
		output.writeln("Y: " + vecY);
		output.writeln("Погрешность относительно точного решения:\n" + MethodError.calcError(exprA, vecX1, vecY1));
		output.writeln("Погрешность методом Рунге-Ромберга:");

		for (int i = 0; i < n; ++i) {
			output.writeln("X[" + i + "]: " + MethodError.rungeRomberg(h, h2, vecY1.get(i), vecY2.get(i * 2), 4.0));
		}

		method.adams(vecX1, vecY1);
		method.setH(h2);
		method.adams(vecX2, vecY2);
		method.setH(h);

		output.writeln("Метод 3: Метод Адамса");
		output.writeln("Решение:");
		output.writeln("X: " + vecX1);
		output.writeln("Y: " + vecY1);
		output.writeln("Точное решение:");
		output.writeln("Y: " + vecY);
		output.writeln("Погрешность относительно точного решения:\n" + MethodError.calcError(exprA, vecX1, vecY1));
		output.writeln("Погрешность методом Рунге-Ромберга:");

		for (int i = 0; i < n; ++i) {
			output.writeln("X[" + i + "]: " + MethodError.rungeRomberg(h, h2, vecY1.get(i), vecY2.get(i * 2), 4.0));
		}

		output.close();
		reader.close();
	}

	public void method2() {
		Reader reader = new Reader("src/data/input/in2.txt");
		Logger output = new Logger("src/data/output/out2.txt");
		ExpTree exprR = new ExpTree(reader.readLine());
		ExpTree exprP = new ExpTree(reader.readLine());
		ExpTree exprQ = new ExpTree(reader.readLine());
		ExpTree exprF = new ExpTree(reader.readLine());
		ExpTree exprA = new ExpTree(reader.readLine());
		double a = reader.readDouble();
		double b = reader.readDouble();
		double h = reader.readDouble();
		double alpha = reader.readDouble();
		double beta = reader.readDouble();
		double delta = reader.readDouble();
		double gamma = reader.readDouble();
		double y0 = reader.readDouble();
		double y1 = reader.readDouble();
		double eps = reader.readDouble();
		double h2 = h / 2.0;
		MethodDiffEqBoundary method = new MethodDiffEqBoundary(exprR, exprP, exprQ, exprF, a, b, h, alpha, beta, delta, gamma, y0, y1);
		int n = method.getN();
		Vector vecX1 = new Vector(n);
		Vector vecY1 = new Vector(n);
		Vector vecX2 = new Vector(n * 2);
		Vector vecY2 = new Vector(n * 2);
		Vector vecY = new Vector(n);
		Vector vecError = new Vector(n);
		Plotter plot = new Plotter(512.0, 512.0);

		method.shooting(vecX1, vecY1, eps);
		method.setH(h2);
		method.shooting(vecX2, vecY2, eps);
		method.setH(h);

		for (int i = 0; i < n; ++i) {
			vecY.set(i, exprA.setVar("x", vecX1.get(i)).calculate());
		}

		output.writeln("Метод 1: Метод стрельбы");
		output.writeln("Решение:");
		output.writeln("X: " + vecX1);
		output.writeln("Y: " + vecY1);
		output.writeln("Точное решение:");
		output.writeln("Y: " + vecY);
		output.writeln("Погрешность относительно точного решения:\n" + MethodError.calcError(exprA, vecX1, vecY1));
		output.writeln("Погрешность методом Рунге-Ромберга:");

		for (int i = 0; i < n; ++i) {
			vecError.set(i, MethodError.rungeRomberg(h, h2, vecY1.get(i), vecY2.get(i * 2), 2.0));
			output.writeln("X[" + i + "]: " + vecError.get(i));
		}

		plot.addData(vecX1, vecY, Color.RED, "Аналитическое");
		plot.addData(vecX1, vecY1, Color.BLUE, "Численное");
		plot.savePng("src/data/plot/plot21.png");

		plot.clearData();
		plot.addData(vecX1, vecError, Color.BLUE, "Погрешность (Рунге-Ромберг)");
		plot.savePng("src/data/plot/plot21_error.png");

		method.finiteDifference(vecX1, vecY1);
		method.setH(h2);
		method.finiteDifference(vecX2, vecY2);
		method.setH(h);

		output.writeln("Метод 2: Метод конечных разностей");
		output.writeln("Решение:");
		output.writeln("X: " + vecX1);
		output.writeln("Y: " + vecY1);
		output.writeln("Точное решение:");
		output.writeln("Y: " + vecY);
		output.writeln("Погрешность относительно точного решения:\n" + MethodError.calcError(exprA, vecX1, vecY1));
		output.writeln("Погрешность методом Рунге-Ромберга:");

		for (int i = 0; i < n; ++i) {
			vecError.set(i, MethodError.rungeRomberg(h, h2, vecY1.get(i), vecY2.get(i * 2), 2.0));
			output.writeln("X[" + i + "]: " + vecError.get(i));
		}

		plot.clearData();
		plot.addData(vecX1, vecY, Color.RED, "Аналитическое");
		plot.addData(vecX1, vecY1, Color.BLUE, "Численное");
		plot.savePng("src/data/plot/plot22.png");

		plot.clearData();
		plot.addData(vecX1, vecError, Color.BLUE, "Погрешность (Рунге-Ромберг)");
		plot.savePng("src/data/plot/plot22_error.png");

		output.close();
		reader.close();
	}
}
