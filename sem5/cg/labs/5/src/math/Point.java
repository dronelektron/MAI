package math;

public class Point {
	public Point(int x, int y) {
		this.x = x;
		this.y = y;
	}

	public void setX(int x) {
		this.x = x;
	}

	public int getX() {
		return x;
	}

	public void setY(int y) {
		this.y = y;
	}

	public int getY() {
		return y;
	}

	public static int getRadius() {
		return RADIUS;
	}

	private static final int RADIUS = 8;
	private int x;
	private int y;
}
