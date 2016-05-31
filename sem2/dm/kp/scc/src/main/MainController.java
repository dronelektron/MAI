package main;

import javafx.fxml.FXML;
import javafx.event.EventType;
import javafx.fxml.Initializable;
import javafx.event.EventHandler;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.TextArea;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;

import main.graph.*;

public class MainController implements Initializable, EventHandler<MouseEvent> {
	@FXML
	public void initialize(URL url, ResourceBundle bundle) {
		canvas = new CustomCanvas(canvasHolder.getPrefWidth(), canvasHolder.getPrefHeight());

		AnchorPane.setBottomAnchor(canvas, 0.0);
		AnchorPane.setTopAnchor(canvas, 0.0);
		AnchorPane.setLeftAnchor(canvas, 0.0);
		AnchorPane.setRightAnchor(canvas, 0.0);

		canvas.setOnMousePressed(this);
		canvas.setOnMouseReleased(this);
		canvas.setOnMouseDragged(this);
		canvasHolder.getChildren().add(canvas);
		graph = new Graph();
		solver = new Solver(graph, textArea);
		curVertex = null;
		curX = -1.0;
		curY = 0.0;
		endX = -1.0;
		endY = 0.0;
		isLeftDragging = false;
		font = Font.font("Arial", FontWeight.BOLD, 16);

		draw();
	}

	@Override
	public void handle(MouseEvent mouseEvent) {
		EventType<? extends MouseEvent> e = mouseEvent.getEventType();
		MouseButton mouseButton = mouseEvent.getButton();
		double ex = mouseEvent.getX();
		double ey = mouseEvent.getY();

		if (e == MouseEvent.MOUSE_PRESSED) {
			curVertex = graph.findVertex(ex, ey);

			if (mouseButton == MouseButton.PRIMARY) {
				graph.addVertex(ex, ey);
				isLeftDragging = true;
			}

			curX = ex;
			curY = ey;
		}
		else if (e == MouseEvent.MOUSE_RELEASED) {
			if (mouseButton == MouseButton.SECONDARY) {
				GraphVertex gv = graph.findVertex(ex, ey);

				if (curVertex != null) {
					if (curVertex == gv) {
						graph.removeVertex(curVertex);
					} else if (gv != null) {
						graph.addEdge(curVertex, gv);
					}
				} else if (gv == null) {
					GraphEdge edge;

					while ((edge = graph.findEdge(curX, curY, ex, ey)) != null) {
						graph.removeEdge(edge);
					}
				}
			} else {
				isLeftDragging = false;
			}

			curVertex = null;
			curX = -1.0;
			curY = 0.0;
			endX = -1.0;
			endY = 0.0;
		}
		else if (e == MouseEvent.MOUSE_DRAGGED) {
			if (mouseButton == MouseButton.PRIMARY) {
				if (curVertex != null) {
					curVertex.setX(ex);
					curVertex.setY(ey);
				}
			}

			endX = ex;
			endY = ey;
		}

		draw();
	}

	@FXML
	public void solveScc() {
		solver.scc();
	}

	@FXML
	public void clear() {
		graph.clear();
		textArea.clear();
		draw();
	}

	private void draw() {
		GraphicsContext gc = canvas.getGraphicsContext2D();
		ArrayList<GraphVertex> vertices = graph.getVertices();
		double radius = GraphVertex.RADIUS;
		double diam = radius * 2.0;
		double smallRadius = radius / 3.0;

		gc.setFill(Color.WHITE);
		gc.setStroke(Color.RED);
		gc.fillRect(0.0, 0.0, canvas.getWidth(), canvas.getHeight());
		gc.setFill(Color.RED);
		gc.setLineWidth(2.0);
		gc.setFont(font);

		for (GraphEdge ge : graph.getEdges()) {
			GraphVertex start = ge.getStart();
			GraphVertex end = ge.getEnd();

			gc.strokeLine(start.getX(), start.getY(), end.getX(), end.getY());
		}

		for (GraphEdge ge : graph.getEdges()) {
			GraphVertex start = ge.getStart();
			GraphVertex end = ge.getEnd();
			double a = end.getX() - start.getX();
			double b = end.getY() - start.getY();
			double c = Math.sqrt(a * a + b * b);
			double cos = a / c;
			double sin = b / c;
			double px = sin;
			double py = -cos;
			double[] xs = {
					end.getX() - diam * cos + px * smallRadius,
					end.getX() - diam * cos - px * smallRadius,
					end.getX() - radius * cos
			};
			double[] ys = {
					end.getY() - diam * sin + py * smallRadius,
					end.getY() - diam * sin - py * smallRadius,
					end.getY() - radius * sin
			};

			gc.fillPolygon(xs, ys, 3);
		}

		for (int i = 0; i < vertices.size(); ++i) {
			GraphVertex gv = vertices.get(i);

			gc.setFill(Color.BLUE);
			gc.fillOval(gv.getX() - radius, gv.getY() - radius, diam, diam);
			gc.setFill(Color.WHITE);
			gc.fillText(String.valueOf(i + 1), gv.getX() - 8, gv.getY() + 4);
		}

		if (!isLeftDragging && curX != -1.0 && endX != -1.0) {
			gc.setStroke(Color.BLACK);
			gc.strokeLine(curX, curY, endX, endY);
		}
	}

	@FXML
	private Canvas canvas;
	@FXML
	private AnchorPane canvasHolder;
	@FXML
	private TextArea textArea;

	private Font font;
	private Graph graph;
	private Solver solver;
	private GraphVertex curVertex;
	private double curX;
	private double curY;
	private double endX;
	private double endY;
	private boolean isLeftDragging;
}
