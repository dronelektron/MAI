package math.splines;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import java.util.LinkedList;
import math.Point;

public class Bezier extends Spline {
	public Bezier(Canvas canvas, LinkedList<Point> points) {
		super(canvas, points);
	}

	@Override
	public void draw() {
		if (points.size() == 0) {
			return;
		}

		GraphicsContext gc = canvas.getGraphicsContext2D();
		double step = 0.01;
		double xPrev = points.get(0).getX();
		double yPrev = points.get(0).getY();

		for (double t = 0; t < step + 1.0; t += step) {
			double x = 0.0;
			double y = 0.0;

			for (int i = 0; i < points.size(); ++i) {
				double basis = getBezierBasis(i, points.size() - 1, t);
				double px = points.get(i).getX();
				double py = points.get(i).getY();

				x += px * basis;
				y += py * basis;
			}

			gc.strokeLine(xPrev, yPrev, x, y);

			xPrev = x;
			yPrev = y;
		}
	}

	private double getBezierBasis(int i, int n, double t) {
		return (fact(n) / (fact(i) * fact(n - i))) * Math.pow(t, i) * Math.pow(1.0 - t, n - i);
	}

	private double fact(int n) {
		return n <= 1 ? 1 : n * fact(n - 1);
	}
}
