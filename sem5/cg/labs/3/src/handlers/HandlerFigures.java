package handlers;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.EventHandler;
import javafx.event.EventType;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Slider;
import javafx.scene.control.TabPane;
import javafx.scene.input.MouseEvent;
import application.figures.*;
import javafx.scene.paint.Color;
import math.Light;
import math.Matrix;

public class HandlerFigures implements ChangeListener<Number>, EventHandler<MouseEvent> {
	public HandlerFigures(Canvas canvas,
			TabPane tabPane,
			Slider[] paramsFig1, Slider[] paramsFig2, Slider[] paramsFig3, Slider[] paramsLight) {
		prevMouseX = 0.0;
		prevMouseY = 0.0;
		figures = new Figure[] {new Figure1(paramsFig1), new Figure2(paramsFig2), new Figure3(paramsFig3)};
		this.canvas = canvas;
		this.tabPane = tabPane;
		this.paramsLight = paramsLight;
		lamp = new Light(new Color(0.0, 0.0, 0.0, 1.0),
				new Color(0.0, 0.0, 0.0, 1.0),
				new Color(0.0, 0.0, 0.0, 1.0),
				30.0);

		for (int i = 0; i < figures.length; ++i) {
			figures[i].generate();
		}

		updateLamp();
		updateFigure(0.0, 0.0);
	}

	@Override
	public void handle(MouseEvent mouseEvent) {
		EventType<? extends MouseEvent> e = mouseEvent.getEventType();

		if (e == MouseEvent.MOUSE_PRESSED) {
			prevMouseX = mouseEvent.getSceneX();
			prevMouseY = mouseEvent.getSceneY();
		}
		else if (e == MouseEvent.MOUSE_DRAGGED) {
			double mouseX = mouseEvent.getSceneX();
			double mouseY = mouseEvent.getSceneY();
			double deltaX = mouseX - prevMouseX;
			double deltaY = mouseY - prevMouseY;

			prevMouseX = mouseX;
			prevMouseY = mouseY;

			updateFigure(deltaY, deltaX);
		}
	}

	@Override
	public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
		int tabId = tabPane.getSelectionModel().getSelectedIndex();

		figures[tabId].generate();

		updateLamp();
		updateFigure(0.0, 0.0);
	}

	private void updateFigure(double deltaAngleX, double deltaAngleY) {
		int tabId = tabPane.getSelectionModel().getSelectedIndex();

		figures[tabId].addAngles(deltaAngleX, deltaAngleY);
		figures[tabId].draw(canvas, lamp);
	}

	private void updateLamp() {
		double ambRed = paramsLight[0].getValue();
		double ambGreen = paramsLight[1].getValue();
		double ambBlue = paramsLight[2].getValue();
		double diffRed = paramsLight[3].getValue();
		double diffGreen = paramsLight[4].getValue();
		double diffBlue = paramsLight[5].getValue();
		double specRed = paramsLight[6].getValue();
		double specGreen = paramsLight[7].getValue();
		double specBlue = paramsLight[8].getValue();

		lamp.setAmbient(new Color(ambRed, ambGreen, ambBlue, 1.0));
		lamp.setDiffuse(new Color(diffRed, diffGreen, diffBlue, 1.0));
		lamp.setSpecular(new Color(specRed, specGreen, specBlue, 1.0));
	}

	private double prevMouseX;
	private double prevMouseY;
	private Canvas canvas;
	private TabPane tabPane;
	private Figure[] figures;
	private Slider[] paramsLight;
	private Light lamp;
}
