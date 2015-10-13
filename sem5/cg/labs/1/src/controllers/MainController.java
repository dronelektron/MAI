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

		Slider[] paramsFig1 = new Slider[] {param1A, param1B, param1C, param1T, param1Step};
		Slider[] paramsFig2 = new Slider[] {param2A, param2B, param2T, param2r, param2R, param2Step};
		HandlerFigures handlerFigures = new HandlerFigures(canvas, tabPane, paramsFig1, paramsFig2);

		for (int i = 0; i < paramsFig1.length; ++i) {
			paramsFig1[i].valueProperty().addListener(handlerFigures);
		}

		for (int i = 0; i < paramsFig2.length; ++i) {
			paramsFig2[i].valueProperty().addListener(handlerFigures);
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
	private Slider param1A;
	@FXML
	private Slider param1B;
	@FXML
	private Slider param1C;
	@FXML
	private Slider param1T;
	@FXML
	private Slider param1Step;

	@FXML
	private Slider param2A;
	@FXML
	private Slider param2B;
	@FXML
	private Slider param2T;
	@FXML
	private Slider param2r;
	@FXML
	private Slider param2R;
	@FXML
	private Slider param2Step;
}
