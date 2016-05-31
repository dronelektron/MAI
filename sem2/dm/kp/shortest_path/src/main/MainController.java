package main;

import javafx.fxml.FXML;
import javafx.event.EventType;
import javafx.fxml.Initializable;
import javafx.event.EventHandler;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;

import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
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
			graph.setCurVertex(graph.findVertex(ex, ey));

			if (mouseButton == MouseButton.PRIMARY) {
				graph.setCurEdge(graph.findEdge(ex, ey));

				if (graph.getCurVertex() == null && graph.getCurEdge() != null) {
					costTextField.setText(String.valueOf(graph.getCurEdge().getCost()));
				} else {
					graph.addVertex(ex, ey);
					isLeftDragging = true;
				}
			}

			curX = ex;
			curY = ey;
		}
		else if (e == MouseEvent.MOUSE_RELEASED) {
			if (mouseButton == MouseButton.SECONDARY) {
				GraphVertex gv = graph.findVertex(ex, ey);

				if (graph.getCurVertex() != null) {
					if (graph.getCurVertex() == gv) {
						graph.removeVertex(graph.getCurVertex());
					} else if (gv != null) {
						graph.addEdge(graph.getCurVertex(), gv);
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

			graph.setCurVertex(null);
			curX = -1.0;
			curY = 0.0;
			endX = -1.0;
			endY = 0.0;
		}
		else if (e == MouseEvent.MOUSE_DRAGGED) {
			if (mouseButton == MouseButton.PRIMARY) {
				GraphVertex cv = graph.getCurVertex();

				if (cv != null) {
					cv.setX(ex);
					cv.setY(ey);
				}
			}

			endX = ex;
			endY = ey;
		}

		draw();
	}

	@FXML
	public void solveShortestPath() {
		solver.shortestPath();
	}

	@FXML
	public void clear() {
		graph.clear();
		textArea.clear();
		draw();
	}

	@FXML
	public void confirmCost() {
		if (costTextField.getText().length() == 0) {
			return;
		}

		GraphEdge gv = graph.getCurEdge();

		if (gv != null) {
			gv.setCost(Integer.parseInt(costTextField.getText()));
			graph.setCurEdge(null);
			draw();
		}
	}

	private void draw() {
		GraphicsContext gc = canvas.getGraphicsContext2D();
		ArrayList<GraphVertex> vertices = graph.getVertices();
		GraphEdge curEdge = graph.getCurEdge();

		gc.setFill(Color.WHITE);
		gc.setStroke(Color.RED);
		gc.fillRect(0.0, 0.0, canvas.getWidth(), canvas.getHeight());
		gc.setFill(Color.RED);
		gc.setLineWidth(2.0);
		gc.setFont(font);

		if (curEdge == null) {
			costTextField.setText("");
		}

		for (GraphEdge ge : graph.getEdges()) {
			GraphVertex start = ge.getStart();
			GraphVertex end = ge.getEnd();

			if (ge == curEdge) {
				continue;
			}

			gc.strokeLine(start.getX(), start.getY(), end.getX(), end.getY());
		}

		if (curEdge != null) {
			GraphVertex start = curEdge.getStart();
			GraphVertex end = curEdge.getEnd();

			gc.setStroke(Color.GREEN);
			gc.strokeLine(start.getX(), start.getY(), end.getX(), end.getY());
		}

		for (GraphEdge ge : graph.getEdges()) {
			if (ge != curEdge) {
				gc.setFill(Color.RED);

				drawArrow(gc, ge);

				gc.setFill(Color.BLACK);
				gc.fillText(String.valueOf(ge.getCost()), ge.getAreaX(), ge.getAreaY());
			}
		}

		if (curEdge != null) {
			gc.setFill(Color.GREEN);

			drawArrow(gc, curEdge);

			gc.setFill(Color.BLACK);
			gc.fillText(String.valueOf(curEdge.getCost()), curEdge.getAreaX(), curEdge.getAreaY());
		}

		for (int i = 0; i < vertices.size(); ++i) {
			GraphVertex gv = vertices.get(i);

			gc.setFill(Color.BLUE);
			gc.fillOval(gv.getX() - GraphVertex.RADIUS, gv.getY() - GraphVertex.RADIUS,
					GraphVertex.RADIUS * 2.0, GraphVertex.RADIUS * 2.0);
			gc.setFill(Color.WHITE);
			gc.fillText(String.valueOf(i + 1), gv.getX() - 8, gv.getY() + 4);
		}

		if (!isLeftDragging && curX != -1.0 && endX != -1.0) {
			gc.setStroke(Color.BLACK);
			gc.strokeLine(curX, curY, endX, endY);
		}
	}

	private void drawArrow(GraphicsContext gc, GraphEdge ge) {
		GraphVertex start = ge.getStart();
		GraphVertex end = ge.getEnd();
		double radius = GraphVertex.RADIUS;
		double smallRadius = radius / 3.0;
		double diam = radius * 2.0;
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

		ge.setAreaX(end.getX() - diam * cos);
		ge.setAreaY(end.getY() - diam * sin);
	}

	@FXML
	private Canvas canvas;
	@FXML
	private AnchorPane canvasHolder;
	@FXML
	private TextArea textArea;
	@FXML
	private TextField costTextField;

	private Font font;
	private Graph graph;
	private Solver solver;
	private double curX;
	private double curY;
	private double endX;
	private double endY;
	private boolean isLeftDragging;
}
