package math.splines;

import javafx.scene.canvas.Canvas;
import java.util.LinkedList;
import math.Point;

public abstract class Spline {
	public Spline(Canvas canvas, LinkedList<Point> points) {
		this.canvas = canvas;
		this.points = points;
	}

	public abstract void draw();

	protected Canvas canvas;
	protected LinkedList<Point> points;
}
