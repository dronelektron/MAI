package application.figures;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Slider;
import java.util.ArrayList;

import javafx.scene.paint.Color;
import math.Matrix;
import math.Vector;

public abstract class Figure {
	public Figure() {
		angleX = 0.0;
		angleY = 0.0;
		mat = new Matrix().initIdentity();
		points = new ArrayList<Vector>();
		indexes = new ArrayList<Integer>();
	}

	public abstract void generate();

	public double getAngleX() {
		return angleX;
	}

	public double getAngleY() {
		return angleY;
	}

	public void setAngles(Matrix projMat, double angleX, double angleY) {
		this.angleX = angleX;
		this.angleY = angleY;

		transform(projMat);
	}

	public void addAngles(Matrix projMat, double deltaAngleX, double deltaAngleY) {
		angleX += deltaAngleX;
		angleY += deltaAngleY;

		transform(projMat);
	}

	public void draw(Canvas canvas) {
		GraphicsContext gc = canvas.getGraphicsContext2D();

		gc.setFill(Color.BLACK);
		gc.fillRect(0.0, 0.0, canvas.getWidth(), canvas.getHeight());
		gc.setStroke(Color.WHITE);

		for (int i = 0; i < indexes.size(); i += 3) {
			Vector p1 = mat.transform(points.get(indexes.get(i))).perspectiveDivide();
			Vector p2 = mat.transform(points.get(indexes.get(i + 1))).perspectiveDivide();
			Vector p3 = mat.transform(points.get(indexes.get(i + 2))).perspectiveDivide();

			if (!inRangeOfCanvas(p1, canvas) ||
					!inRangeOfCanvas(p2, canvas) ||
					!inRangeOfCanvas(p3, canvas)) {
				continue;
			}

			drawTriangle(gc, p1, p2, p3);
		}
	}

	private void transform(Matrix projMat) {
		Matrix rotMatX = new Matrix().initRotationX(angleX);
		Matrix rotMatY = new Matrix().initRotationY(angleY);

		mat = projMat.mul(rotMatX.mul(rotMatY));
	}

	private void drawTriangle(GraphicsContext gc, Vector p1, Vector p2, Vector p3) {
		if (p1.areaTimesTwo(p2, p3) <= 0.0) {
			return;
		}

		gc.strokeLine(p1.getX(), p1.getY(), p2.getX(), p2.getY());
		gc.strokeLine(p2.getX(), p2.getY(), p3.getX(), p3.getY());
		gc.strokeLine(p3.getX(), p3.getY(), p1.getX(), p1.getY());
	}

	private boolean inRangeOfCanvas(Vector p, Canvas canvas) {
		return p.getX() >= 0.0 && p.getX() < canvas.getWidth() &&
				p.getY() >= 0.0 && p.getY() < canvas.getHeight();
	}

	private double angleX;
	private double angleY;
	protected Slider[] params;
	protected Matrix mat;
	protected ArrayList<Vector> points;
	protected ArrayList<Integer> indexes;
}
