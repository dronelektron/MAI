package controllers;

import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Slider;
import javafx.scene.control.TabPane;
import javafx.scene.layout.AnchorPane;
import application.CustomCanvas;
import handlers.HandlerFigures;

public class MainController {
	@FXML
	private void initialize() {
		Canvas canvas = new CustomCanvas(canvasHolder.getPrefWidth(), canvasHolder.getPrefHeight());

		AnchorPane.setBottomAnchor(canvas, 0.0);
		AnchorPane.setTopAnchor(canvas, 0.0);
		AnchorPane.setLeftAnchor(canvas, 0.0);
		AnchorPane.setRightAnchor(canvas, 0.0);

		Slider[] paramsFig1 = new Slider[] {param1Height, param1Radius, param1Sides};
		Slider[] paramsFig2 = new Slider[] {param2XStart, param2XEnd, param2YStart, param2YEnd, param2A, param2XStep, param2YStep};
		Slider[] paramsFig3 = new Slider[] {param3Height, param3Radius, param3Sides, param3Angle};
		HandlerFigures handlerFigures = new HandlerFigures(canvas, tabPane, paramsFig1, paramsFig2, paramsFig3);

		for (int i = 0; i < paramsFig1.length; ++i) {
			paramsFig1[i].valueProperty().addListener(handlerFigures);
		}

		for (int i = 0; i < paramsFig2.length; ++i) {
			paramsFig2[i].valueProperty().addListener(handlerFigures);
		}

		for (int i = 0; i < paramsFig3.length; ++i) {
			paramsFig3[i].valueProperty().addListener(handlerFigures);
		}

		canvas.setOnMousePressed(handlerFigures);
		canvas.setOnMouseDragged(handlerFigures);
		canvasHolder.getChildren().add(canvas);
	}

	@FXML
	private AnchorPane canvasHolder;
	@FXML
	private TabPane tabPane;

	@FXML
	private Slider param1Height;
	@FXML
	private Slider param1Radius;
	@FXML
	private Slider param1Sides;

	@FXML
	private Slider param2XStart;
	@FXML
	private Slider param2XEnd;
	@FXML
	private Slider param2YStart;
	@FXML
	private Slider param2YEnd;
	@FXML
	private Slider param2A;
	@FXML
	private Slider param2XStep;
	@FXML
	private Slider param2YStep;

	@FXML
	private Slider param3Height;
	@FXML
	private Slider param3Radius;
	@FXML
	private Slider param3Sides;
	@FXML
	private Slider param3Angle;
}
