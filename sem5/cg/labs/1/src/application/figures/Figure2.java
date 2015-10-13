package application.figures;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Slider;
import javafx.scene.paint.Color;
import math.Vector;

public class Figure2 extends Figure {
	public Figure2(Slider[] params) {
		super();

		this.params = params;
	}

	@Override
	public void draw(Canvas canvas) {
		Slider paramA = params[0];
		Slider paramB = params[1];
		Slider paramT = params[2];
		Slider paramr = params[3];
		Slider paramR = params[4];
		Slider paramStep =  params[5];
		GraphicsContext gc = canvas.getGraphicsContext2D();

		gc.setFill(Color.BLACK);
		gc.fillRect(0.0, 0.0, canvas.getWidth(), canvas.getHeight());
		gc.setStroke(Color.WHITE);

		Vector prevPoint = new Vector(0.0, 0.0, 0.0, 1.0);

		prevPoint = mat.transform(prevPoint).perspectiveDivide();

		for (double t = -paramT.getValue(); t <= paramT.getValue(); t += paramStep.getValue()) {

			double r = paramr.getValue();
			double R = paramR.getValue();
			double x = paramA.getValue() * Math.sin(t);
			double y = paramB.getValue() * t;
			double expr1 = R - Math.sqrt(x * x + y * y);
			double expr2 = r * r - Math.pow(expr1, 2.0);

			if (expr1 < 0.0 || expr2 < 0.0) {
				continue;
			}

			double z = Math.sqrt(expr2);

			Vector nextPoint = new Vector(x, y, z, 1.0);

			nextPoint = mat.transform(nextPoint).perspectiveDivide();

			if (nextPoint.getX() < 0.0 ||
					nextPoint.getY() < 0 ||
					nextPoint.getX() >= canvas.getWidth() ||
					nextPoint.getY() >= canvas.getHeight()) {
				continue;
			}

			gc.strokeLine(prevPoint.getX(), prevPoint.getY(), nextPoint.getX(), nextPoint.getY());
			prevPoint.copy(nextPoint);
		}
	}

	private Slider[] params;
}
