package handlers;

import javafx.event.EventHandler;
import javafx.event.EventType;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import java.util.LinkedList;
import javafx.scene.paint.Color;
import math.Point;
import math.splines.*;

public class MainHandler implements EventHandler<MouseEvent> {
	public MainHandler(Canvas canvas) {
		points = new LinkedList<>();
		curPoint = null;
		this.canvas = canvas;

		splines = new Spline[] {
				new Bezier(canvas, points),
				new Lagrange(canvas, points),
				new Cardinal(canvas, points)
		};

		splineIndex = 0;
		viewOption1 = false;
	}

	public void setSplineIndex(int index) {
		splineIndex = index;

		draw();
	}

	public void setViewOption1(boolean active) {
		viewOption1 = active;

		draw();
	}

	@Override
	public void handle(MouseEvent mouseEvent) {
		EventType<? extends MouseEvent> e = mouseEvent.getEventType();
		MouseButton mouseButton = mouseEvent.getButton();
		int ex = (int)mouseEvent.getX();
		int ey = (int)mouseEvent.getY();

		if (e == MouseEvent.MOUSE_PRESSED) {
			Point p = findPoint(ex, ey);

			if (mouseButton == MouseButton.PRIMARY) {
				if (p == null) {
					points.add(new Point(ex, ey));
				} else {
					curPoint = p;
				}
			} else if (p != null) {
				points.remove(p);
			}
		}
		else if (e == MouseEvent.MOUSE_RELEASED) {
			curPoint = null;
		}
		else if (e == MouseEvent.MOUSE_DRAGGED) {
			if (curPoint != null) {
				curPoint.setX(ex);
				curPoint.setY(ey);
			}
		}

		draw();
	}

	private Point findPoint(int x, int y) {
		for (Point p : points) {
			int px = p.getX();
			int py = p.getY();
			int dx = px - x;
			int dy = py - y;
			int r = Point.getRadius();

			if (dx * dx + dy * dy <= r * r) {
				return p;
			}
		}

		return null;
	}

	private void draw() {
		GraphicsContext gc = canvas.getGraphicsContext2D();
		int radius = Point.getRadius();
		int diam = radius * 2;

		gc.setFill(Color.BLACK);
		gc.setStroke(Color.WHITE);
		gc.fillRect(0.0, 0.0, canvas.getWidth(), canvas.getHeight());

		for (Point p : points) {
			gc.strokeOval(p.getX() - radius, p.getY() - radius, diam, diam);
		}

		if (viewOption1) {
			gc.setStroke(Color.RED);

			for (int i = 1; i < points.size(); ++i) {
				Point p1 = points.get(i - 1);
				Point p2 = points.get(i);

				gc.strokeLine(p1.getX(), p1.getY(), p2.getX(), p2.getY());
			}
		}

		gc.setStroke(Color.WHITE);

		splines[splineIndex].draw();
	}

	private LinkedList<Point> points;
	private Point curPoint;
	private Canvas canvas;
	private Spline[] splines;
	private int splineIndex;
	private boolean viewOption1;
}
