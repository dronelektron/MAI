package controllers;

import javafx.fxml.FXML;
import handlers.MainHandler;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.RadioMenuItem;
import javafx.scene.layout.AnchorPane;
import application.CustomCanvas;

public class MainController {
	@FXML
	private void initialize() {
		Canvas canvas = new CustomCanvas(canvasHolder.getPrefWidth(), canvasHolder.getPrefHeight());

		AnchorPane.setBottomAnchor(canvas, 0.0);
		AnchorPane.setTopAnchor(canvas, 0.0);
		AnchorPane.setLeftAnchor(canvas, 0.0);
		AnchorPane.setRightAnchor(canvas, 0.0);

		mainHandler = new MainHandler(canvas);
		canvas.setOnMousePressed(mainHandler);
		canvas.setOnMouseReleased(mainHandler);
		canvas.setOnMouseDragged(mainHandler);
		canvasHolder.getChildren().add(canvas);
	}

	@FXML
	private void menuSpline1() {
		mainHandler.setSplineIndex(0);
	}

	@FXML
	private void menuSpline2() {
		mainHandler.setSplineIndex(1);
	}

	@FXML
	private void menuSpline3() {
		mainHandler.setSplineIndex(2);
	}

	@FXML
	private void menuView1() {
		mainHandler.setViewOption1(viewItem1.isSelected());
	}

	private MainHandler mainHandler;

	@FXML
	private AnchorPane canvasHolder;
	@FXML
	private RadioMenuItem viewItem1;
}
