package application.figures;

import javafx.scene.canvas.Canvas;
import math.Matrix;

public abstract class Figure {
	public Figure() {
		angleX = 0.0;
		angleY = 0.0;
		mat = new Matrix().initIdentity();
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

	public abstract void draw(Canvas canvas);

	private double angleX;
	private double angleY;
	protected Matrix mat;
}
