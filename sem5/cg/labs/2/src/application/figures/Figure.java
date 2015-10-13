package application.figures;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Slider;
import java.util.ArrayList;
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

	private void transform(Matrix projMat) {
		Matrix rotMatX = new Matrix().initRotationX(angleX);
		Matrix rotMatY = new Matrix().initRotationY(angleY);

		mat = projMat.mul(rotMatX.mul(rotMatY));
	}

	protected void drawTriangle(GraphicsContext gc, Vector p1, Vector p2, Vector p3) {
		if (p1.areaTimesTwo(p2, p3) <= 0.0) {
			return;
		}

		gc.strokeLine(p1.getX(), p1.getY(), p2.getX(), p2.getY());
		gc.strokeLine(p2.getX(), p2.getY(), p3.getX(), p3.getY());
		gc.strokeLine(p3.getX(), p3.getY(), p1.getX(), p1.getY());
	}

	protected boolean inRange(double x1, double y1, double x2, double y2, double x, double y) {
		return !(x < x1 || x > x2 || y < y1 || y > y2);
	}

	public abstract void generate();
	public abstract void draw(Canvas canvas);

	private double angleX;
	private double angleY;
	protected Slider[] params;
	protected Matrix mat;
	protected ArrayList<Vector> points;
	protected ArrayList<Integer> indexes;
}
