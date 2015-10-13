package math;

public class Vector {
	public Vector() {
		x = 0.0;
		y = 0.0;
		z = 0.0;
		w = 0.0;
	}

	public Vector(double x, double y, double z, double w) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}

	public Vector copy(Vector v) {
		x = v.x;
		y = v.y;
		z = v.z;
		w = v.w;

		return this;
	}

	public void setX(double x) {
		this.x = x;
	}

	public double getX() {
		return x;
	}

	public void setY(double y) {
		this.y = y;
	}

	public double getY() {
		return y;
	}

	public void setZ(double z) {
		this.z = z;
	}

	public double getZ() {
		return z;
	}

	public void setW(double w) {
		this.w = w;
	}

	public double getW() {
		return w;
	}

	public double areaTimesTwo(Vector a, Vector b) {
		double x1 = a.x - x;
		double y1 = a.y - y;
		double x2 = b.x - x;
		double y2 = b.y - y;

		return x1 * y2 - x2 * y1;
	}

	public Vector add(Vector v) {
		return new Vector(x + v.x, y + v.y, z + v.z, w + v.w);
	}

	public Vector mul(double a) {
		return new Vector(x * a, y * a, z * a, w * a);
	}

	public Vector perspectiveDivide() {
		return new Vector(x / w, y / w, z / w, 1.0);
	}

	private double x;
	private double y;
	private double z;
	private double w;
}
