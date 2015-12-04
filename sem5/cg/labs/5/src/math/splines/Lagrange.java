package math.splines;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import java.util.LinkedList;
import math.Point;

public class Lagrange extends Spline {
	public Lagrange(Canvas canvas, LinkedList<Point> points) {
		super(canvas, points);
	}

	@Override
	public void draw() {
		GraphicsContext gc = canvas.getGraphicsContext2D();
		double xPrev = 0.0;
		double yPrev = 0.0;

		for (double x = 0; x < canvas.getWidth(); x += 1.0) {
			double y = 0.0;

			for (int i = 0; i < points.size(); ++i) {
				double s = 1.0;
				double t = 1.0;
				double pyi = points.get(i).getY();

				for (int j = 0; j < points.size(); ++j) {
					if (i != j) {
						double pxi = points.get(i).getX();
						double pxj = points.get(j).getX();

						s = s * (x - pxj);
						t = t * (pxi - pxj);
					}
				}

				if (t == 0.0) {
					t = 1.0;
				}

				y += (s / t) * pyi;
			}

			if (y < 0.0) {
				y = 0.0;
			} else if (y > canvas.getHeight()) {
				y = canvas.getHeight() - 1.0;
			}

			gc.strokeLine(xPrev, yPrev, x, y);

			xPrev = x;
			yPrev = y;
		}
	}
}
