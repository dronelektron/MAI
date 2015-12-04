package math.splines;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import java.util.LinkedList;
import math.Point;

public class Cardinal extends Spline {
	public Cardinal(Canvas canvas, LinkedList<Point> points) {
		super(canvas, points);
	}

	@Override
	public void draw() {
		if (points.size() < 4) {
			return;
		}

		GraphicsContext gc = canvas.getGraphicsContext2D();
		double step = 0.01;
		double a = 0.5;

		for (int i = 0; i <= points.size() - 4; ++i) {
			Point p0 = points.get(i);
			Point p1 = points.get(i + 1);
			Point p2 = points.get(i + 2);
			Point p3 = points.get(i + 3);
			double xPrev = p1.getX();
			double yPrev = p1.getY();

			for (double s = 0; s < step + 1.0; s += step) {
				double h1 = 2.0 * Math.pow(s, 3.0) - 3.0 * Math.pow(s, 2.0) + 1.0;
				double h2 = -2.0 * Math.pow(s, 3.0) + 3.0 * Math.pow(s, 2.0);
				double h3 = Math.pow(s, 3.0) - 2.0 * Math.pow(s, 2.0) + s;
				double h4 = Math.pow(s, 3.0) - Math.pow(s, 2.0);
				double t1x = a * (p2.getX() - p0.getX());
				double t1y = a * (p2.getY() - p0.getY());
				double t2x = a * (p3.getX() - p1.getX());
				double t2y = a * (p3.getY() - p1.getY());
				double x = h1 * p1.getX() + h2 * p2.getX() + h3 * t1x + h4 * t2x;
				double y = h1 * p1.getY() + h2 * p2.getY() + h3 * t1y + h4 * t2y;

				gc.strokeLine(xPrev, yPrev, x, y);

				xPrev = x;
				yPrev = y;
			}
		}
	}
}
