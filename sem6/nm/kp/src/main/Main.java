package main;

import java.awt.*;

import libnm.math.Vector;
import util.Plotter;

public class Main {
	public static void main(String[] args) {
		double a1 = 1.0;
		double a2 = 2.0;
		double a3 = 15.0;
		double y0 = 0.3;
		double a = 0.0;
		double b = 300.0;
		double h = 0.01;
		int n = (int)((b - a) / h) + 1;
		Vector vecX = new Vector(n);
		Vector vecY = new Vector(n);
		Plotter plotter = new Plotter(1024.0, 512.0);
		Euler method = new Euler(a1, a2, a3, y0, a, b, h);

		method.solve(vecX, vecY);

		plotter.addData(vecX, vecY, Color.RED, "Y");
		plotter.savePng("src/data/plot.png");
	}
}
