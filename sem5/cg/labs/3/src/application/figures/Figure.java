package application.figures;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Slider;
import javafx.scene.paint.Color;
import java.util.ArrayList;
import math.Light;
import math.Matrix;
import math.Vector;

public abstract class Figure {
	public Figure() {
		mat = new Matrix().initIdentity();
		points = new ArrayList<>();
		indexes = new ArrayList<>();
	}

	public abstract void generate();

	public void addAngles(double deltaAngleX, double deltaAngleY) {
		if (deltaAngleX != 0.0) {
			mat = new Matrix().initRotationX(deltaAngleX).mul(mat);
		} else {
			mat = new Matrix().initRotationY(deltaAngleY).mul(mat);
		}
	}

	public void draw(Canvas canvas, Light lamp) {
		final double WIDTH = canvas.getWidth();
		final double HEIGHT = canvas.getHeight();
		final double DEEP = 1001;

		GraphicsContext gc = canvas.getGraphicsContext2D();
		Matrix viewMat = new Matrix().initTranslation(0.0, 0.0, DEEP);
		Matrix projMat = new Matrix().initPerspective(70.0, WIDTH / HEIGHT);
		Matrix screenMat = new Matrix().initScreenSpace(WIDTH, HEIGHT);
		Matrix totalMat = screenMat.mul(projMat.mul(viewMat));

		gc.setFill(Color.BLACK);
		gc.fillRect(0.0, 0.0, canvas.getWidth(), canvas.getHeight());

		for (int i = 0; i < indexes.size(); i += 3) {
			Vector p1 = mat.transform(points.get(indexes.get(i)));
			Vector p2 = mat.transform(points.get(indexes.get(i + 1)));
			Vector p3 = mat.transform(points.get(indexes.get(i + 2)));
			Vector normal = p1.sub(p2).vec(p3.sub(p2)).mul(-1.0);
			Vector light = new Vector(0.0, 0.0, -DEEP, 1.0).sub(p2);
			Vector eye = new Vector(0.0, 0.0, -DEEP, 1.0).sub(p2);
			Color color = lamp.getColor(normal, light, eye);

			p1 = totalMat.transform(p1).perspectiveDivide();
			p2 = totalMat.transform(p2).perspectiveDivide();
			p3 = totalMat.transform(p3).perspectiveDivide();

			if (!inRangeOfCanvas(p1, canvas) ||
					!inRangeOfCanvas(p2, canvas) ||
					!inRangeOfCanvas(p3, canvas)) {
				continue;
			}

			drawFilledTriangle(gc, p1, p2, p3, color);
		}
	}

	private void drawFilledTriangle(GraphicsContext gc, Vector v1, Vector v2, Vector v3, Color color) {
		if (v1.areaTimesTwo(v2, v3) <= 0.0) {
			return;
		}

		gc.setFill(color);
		gc.fillPolygon(new double[] {v1.getX(), v2.getX(), v3.getX()},
				new double[] {v1.getY(), v2.getY(), v3.getY()}, 3);
	}

	private boolean inRangeOfCanvas(Vector p, Canvas canvas) {
		return p.getX() >= 0.0 && p.getX() < canvas.getWidth() &&
				p.getY() >= 0.0 && p.getY() < canvas.getHeight();
	}

	protected Matrix mat;
	protected Slider[] params;
	protected ArrayList<Vector> points;
	protected ArrayList<Integer> indexes;
}
