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
		Slider[] paramsFig2 = new Slider[] {param2R, param2r, param2RStep, param2rStep};
		Slider[] paramsFig3 = new Slider[] {param3Radius, param3Angle, param3StepV, param3StepH};
		Slider[] paramsLight = new Slider[] {
				paramLightARed, paramLightAGreen, paramLightABlue,
				paramLightDRed, paramLightDGreen, paramLightDBlue,
				paramLightSRed, paramLightSGreen, paramLightSBlue
		};

		HandlerFigures handlerFigures = new HandlerFigures(canvas, tabPane, paramsFig1, paramsFig2, paramsFig3, paramsLight);

		for (int i = 0; i < paramsFig1.length; ++i) {
			paramsFig1[i].valueProperty().addListener(handlerFigures);
		}

		for (int i = 0; i < paramsFig2.length; ++i) {
			paramsFig2[i].valueProperty().addListener(handlerFigures);
		}

		for (int i = 0; i < paramsFig3.length; ++i) {
			paramsFig3[i].valueProperty().addListener(handlerFigures);
		}

		for (int i = 0; i < paramsLight.length; ++i) {
			paramsLight[i].valueProperty().addListener(handlerFigures);
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
	private Slider param2R;
	@FXML
	private Slider param2r;
	@FXML
	private Slider param2RStep;
	@FXML
	private Slider param2rStep;

	@FXML
	private Slider param3Radius;
	@FXML
	private Slider param3Angle;
	@FXML
	private Slider param3StepV;
	@FXML
	private Slider param3StepH;

	@FXML
	private Slider paramLightARed;
	@FXML
	private Slider paramLightAGreen;
	@FXML
	private Slider paramLightABlue;
	@FXML
	private Slider paramLightDRed;
	@FXML
	private Slider paramLightDGreen;
	@FXML
	private Slider paramLightDBlue;
	@FXML
	private Slider paramLightSRed;
	@FXML
	private Slider paramLightSGreen;
	@FXML
	private Slider paramLightSBlue;
}
