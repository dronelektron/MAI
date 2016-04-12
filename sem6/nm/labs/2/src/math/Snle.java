package math;

import libnm.math.expression.ExpTree;
import libnm.math.method.MethodSnle;
import libnm.math.*;
import libnm.util.*;

public class Snle {
	public Snle() {
		m_method = new MethodSnle();
	}

	public void method11() {
		m_method1("src/data/input/in11.txt", "src/data/logs/log11.txt", "src/data/output/out11.txt");
	}

	public void method12() {
		m_method2("src/data/input/in12.txt", "src/data/logs/log12.txt", "src/data/output/out12.txt");
	}

	public void method21() {
		m_method1("src/data/input/in21.txt", "src/data/logs/log21.txt", "src/data/output/out21.txt");
	}

	public void method22() {
		m_method2("src/data/input/in22.txt", "src/data/logs/log22.txt", "src/data/output/out22.txt");
	}

	private void m_method1(String inputName, String loggerName, String outputName) {
		Reader reader = new Reader(inputName);
		Logger logger = new Logger(loggerName);
		Logger output = new Logger(outputName);
		int n = reader.readInt();
		double a = reader.readDouble();
		double q = reader.readDouble();
		double eps = reader.readDouble();
		Vector vec = reader.readVector();
		Vector vecX = new Vector(n);
		ExpTree[] exprs = new ExpTree[n];

		output.writeln("Метод 1: Метод простой итерации");

		for (int i = 0; i < n; ++i) {
			exprs[i] = new ExpTree(reader.readLine());
			exprs[i].setVar("a", a);
		}

		m_method.setLogger(logger);
		m_method.simpleIteration(exprs, vec, vecX, eps, q);

		output.writeln("Решение: " + vecX);
		output.close();
		logger.close();
		reader.close();
	}

	private void m_method2(String inputName, String loggerName, String outputName) {
		Reader reader = new Reader(inputName);
		Logger logger = new Logger(loggerName);
		Logger output = new Logger(outputName);
		int n = reader.readInt();
		double a = reader.readDouble();
		double eps = reader.readDouble();
		Vector vec = reader.readVector();
		Vector vecX = new Vector(n);
		ExpTree[] exprs = new ExpTree[n];

		output.writeln("Метод 2: Метод Ньютона");

		for (int i = 0; i < n; ++i) {
			exprs[i] = new ExpTree(reader.readLine());
			exprs[i].setVar("a", a);
		}

		m_method.setLogger(logger);
		m_method.newton(exprs, vec, vecX, eps);

		output.writeln("Решение: " + vecX);
		output.close();
		logger.close();
		reader.close();
	}

	private MethodSnle m_method;
}
