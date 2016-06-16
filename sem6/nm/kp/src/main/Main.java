package main;

import java.awt.*;

import libnm.math.Vector;
import libnm.util.Logger;
import util.Plotter;

public class Main {
	public static void main(String[] args) {
		double[][] paramsA = {
				// {a1, a3}
				// {a2, ...}
				// Задание 1
				{1.0, 0.0},
				{0.0, 0.5, 1.0, 1.5, 2.0},
				// Задание 2 (а)
				{1.0, 15.0},
				{0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1},
				// Задание 2 (б)
				{1.5, 10.0},
				{0.5, 0.8, 1.0, 1.3, 1.5, 1.7, 2.0, 2.1, 2.3, 2.5, 2.6, 3.0, 3.3, 3.5, 3.6},
				// Экстремумы
				{1.0, 15.0},
				{0.5, 0.9, 1.4},
				// При увеличении a1 (в 2 раза)
				{2.0, 15.0},
				{0.5, 0.9, 1.4},
				// При уменьшении a3 (в 3 раза)
				{1.0, 5.0},
				{0.5, 0.9, 1.4}
		};
		double y0 = 0.3;
		double a = 0.0;
		double b = 300.0;
		double h = 0.01;
		int n = (int)((b - a) / h) + 1;
		Vector vecX = new Vector(n);
		Vector vecY = new Vector(n);

		for (int i = 0; i < paramsA.length; i += 2) {
			double a1 = paramsA[i][0];
			double a3 = paramsA[i][1];

			for (int j = 0; j < paramsA[i + 1].length; ++j) {
				double a2 = paramsA[i + 1][j];
				Logger logger = new Logger("src/data/logs/log_" + (i / 2 + 1) + "_" + (j + 1) + ".txt");
				Plotter plotter = new Plotter(1024.0, 512.0);
				Euler method = new Euler(a1, a2, a3, y0, a, b, h);

				method.setLogger(logger);
				method.solve(vecX, vecY);

				plotter.addData(vecX, vecY, Color.RED, "Y");
				plotter.savePng("src/data/plot_" + (i / 2 + 1) + "_" + (j + 1) + "_a1=" + a1 + "_a2=" + a2 + "_a3=" + a3 + ".png");

				logger.close();
			}
		}
	}
}
