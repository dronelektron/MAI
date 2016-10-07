package math;

import libnm.math.method.MethodSle;
import libnm.math.*;
import libnm.util.*;

public class Sle {
	public Sle() {
		m_method = new MethodSle();
	}

	public void method1() {
		Reader reader = new Reader("src/data/input/in1.txt");
		Logger logger = new Logger("src/data/logs/log1.txt");
		Logger output = new Logger("src/data/output/out1.txt");
		int n = reader.readInt();
		Matrix mat = reader.readMatrix(n);
		Vector vec = reader.readVector();
		Vector vecX = new Vector(n);
		Matrix matInv = new Matrix(n);

		output.writeln("Метод 1: LUP - разложение");
		output.writeln("Исходная матрица:\n" + mat);
		output.writeln("Исходный вектор:\n" + vec);

		m_method.setLogger(logger);

		if (!m_method.lup(mat, vec, vecX)) {
			logger.writeln("Матрица A является вырожденной");
		} else {
			m_method.matInverse(mat, matInv);

			output.writeln("Решение:\n" + vecX);
			output.writeln("Обратная матрица:\n" + matInv);
			output.writeln("Определитель:\n" + m_method.matDet(mat));
		}

		output.close();
		logger.close();
		reader.close();
	}

	public void method2() {
		Reader reader = new Reader("src/data/input/in2.txt");
		Logger logger = new Logger("src/data/logs/log2.txt");
		Logger output = new Logger("src/data/output/out2.txt");
		int n = reader.readInt();
		Matrix mat = reader.readMatrix(n);
		Vector vec = reader.readVector();
		Vector vecX = new Vector(n);

		output.writeln("Метод 2: Метод прогонки");
		output.writeln("Исходная матрица:\n" + mat);
		output.writeln("Исходный вектор:\n" + vec);

		m_method.setLogger(logger);

		if (!m_method.tma(mat, vec, vecX, true)) {
			logger.writeln("Матрица A не является трехдиагональной или не выполнено условие |b| >= |a| + |b|");
		} else {
			output.writeln("Решение:\n" + vecX);
		}

		output.close();
		logger.close();
		reader.close();
	}

	public void method3() {
		Reader reader = new Reader("src/data/input/in3.txt");
		Logger logger = new Logger("src/data/logs/log3.txt");
		Logger output = new Logger("src/data/output/out3.txt");
		int n = reader.readInt();
		double eps = reader.readDouble();
		Matrix mat = reader.readMatrix(n);
		Vector vec = reader.readVector();
		Vector vecX = new Vector(n);

		output.writeln("Метод 3: Метод простых итераций");
		output.writeln("Исходная матрица:\n" + mat);
		output.writeln("Исходный вектор:\n" + vec);

		m_method.setLogger(logger);
		m_method.simpleIteration(mat, vec, vecX, eps);

		output.writeln("Решение:\n" + vecX);
		output.close();
		logger.close();
		reader.close();
	}

	public void method4() {
		Reader reader = new Reader("src/data/input/in3.txt");
		Logger logger = new Logger("src/data/logs/log4.txt");
		Logger output = new Logger("src/data/output/out4.txt");
		int n = reader.readInt();
		double eps = reader.readDouble();
		Matrix mat = reader.readMatrix(n);
		Vector vec = reader.readVector();
		Vector vecX = new Vector(n);

		output.writeln("Метод 4: Метод Зейделя");
		output.writeln("Исходная матрица:\n" + mat);
		output.writeln("Исходный вектор:\n" + vec);

		m_method.setLogger(logger);
		m_method.seidel(mat, vec, vecX, eps);

		output.writeln("Решение:\n" + vecX);
		output.close();
		logger.close();
		reader.close();
	}

	public void method5() {
		Reader reader = new Reader("src/data/input/in4.txt");
		Logger logger = new Logger("src/data/logs/log5.txt");
		Logger output = new Logger("src/data/output/out5.txt");
		int n = reader.readInt();
		double eps = reader.readDouble();
		Matrix mat = reader.readMatrix(n);
		Matrix matX = new Matrix(n);
		Vector vecX = new Vector(n);

		output.writeln("Метод 5: Метод вращений");
		output.writeln("Исходная матрица:\n" + mat);

		m_method.setLogger(logger);

		if (!m_method.rotation(mat, matX, vecX, eps)) {
			logger.writeln("Матрица А не является симметричной");
		} else {
			output.writeln("Собственные значения:");

			for (int i = 0; i < n; ++i) {
				output.writeln("Lambda #" + (i + 1) + ": " + vecX.get(i));
			}

			output.writeln("Матрица собственных векторов:\n" + matX);
		}

		output.close();
		logger.close();
		reader.close();
	}

	public void method6() {
		Reader reader = new Reader("src/data/input/in5.txt");
		Logger logger = new Logger("src/data/logs/log6.txt");
		Logger output = new Logger("src/data/output/out6.txt");
		int n = reader.readInt();
		double eps = reader.readDouble();
		Matrix mat = reader.readMatrix(n);
		Complex[] res = new Complex[n];

		output.writeln("Метод 6: QR - алгоритм");
		output.writeln("Исходная матрица:\n" + mat);

		for (int i = 0; i < n; ++i) {
			res[i] = new Complex(0.0, 0.0);
		}

		m_method.setLogger(logger);
		m_method.qr(mat, res, eps);

		output.writeln("Собственные значения:");

		for (int i = 0; i < n; ++i) {
			output.writeln("Lambda #" + (i + 1) + ": " + res[i]);
		}

		output.close();
		logger.close();
		reader.close();
	}

	private MethodSle m_method;
}
