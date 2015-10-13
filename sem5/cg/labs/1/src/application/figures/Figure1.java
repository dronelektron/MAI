package application.figures;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Slider;
import javafx.scene.paint.Color;
import math.Vector;

public class Figure1 extends Figure {
	public Figure1(Slider[] params) {
		super();

		this.params = params;
	}

	@Override
	public void draw(Canvas canvas) {
		Slider paramA = params[0];
		Slider paramB = params[1];
		Slider paramC = params[2];
		Slider paramT = params[3];
		Slider paramStep =  params[4];
		GraphicsContext gc = canvas.getGraphicsContext2D();
		Vector prevPoint = new Vector(paramA.getValue(), 0.0, 0.0, 1.0);

		prevPoint = mat.transform(prevPoint).perspectiveDivide();

		gc.setFill(Color.BLACK);
		gc.fillRect(0.0, 0.0, canvas.getWidth(), canvas.getHeight());
		gc.setStroke(Color.WHITE);

		for (double t = 0.0; t <= paramT.getValue(); t += paramStep.getValue()) {
			Vector nextPoint = new Vector(paramA.getValue() * Math.cos(t),
					paramB.getValue() * Math.sin(t),
					paramC.getValue() * t, 1.0);

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
