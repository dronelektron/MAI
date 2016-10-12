package main;

import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;

import javafx.fxml.FXML;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.fxml.Initializable;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;

import libnm.math.Matrix;
import libnm.math.Vector;
import libnm.math.expression.ExpTree;
import libnm.math.pde.Parabolic2D;
import libnm.util.Logger;

public class MainController implements Initializable, ChangeListener<Number> {
	@Override
	public void initialize(URL location, ResourceBundle resources) {
		sliceType = SLICE_XY;
		method = new Parabolic2D();
		matU = new ArrayList<>();
		vecX = new Vector();
		vecY = new Vector();
		vecT = new Vector();

		scrollBarK1.valueProperty().addListener(this);
		scrollBarK2.valueProperty().addListener(this);
	}

	@Override
	public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
		updatePlotters();
	}

	public void buttonSolve(ActionEvent actionEvent) {
		int methodType;
		Logger output = new Logger("output.txt");
		RadioButton rbMethod = (RadioButton)rbGroupMethod.getSelectedToggle();

		if (rbMethod == radioButtonADI) {
			methodType = Parabolic2D.METHOD_ALTERNATING_DIRECTION;
		} else {
			methodType = Parabolic2D.METHOD_FRACTIONAL_STEP;
		}

		/* TEST
		fieldA.setText("1");
		fieldB.setText("1");
		fieldExprF.setText("0");
		fieldExprPsi.setText("cos(x)*cos(y)");
		fieldAlpha1.setText("0");
		fieldBeta1.setText("1");
		fieldAlpha2.setText("0");
		fieldBeta2.setText("1");
		fieldAlpha3.setText("0");
		fieldBeta3.setText("1");
		fieldAlpha4.setText("0");
		fieldBeta4.setText("1");
		fieldExprFi1.setText("cos(y)*e^(-2*a*t)");
		fieldExprFi2.setText("-cos(y)*e^(-2*a*t)");
		fieldExprFi3.setText("cos(x)*e^(-2*a*t)");
		fieldExprFi4.setText("-cos(x)*e^(-2*a*t)");
		fieldLx.setText("3.14");
		fieldLy.setText("3.14");
		fieldNx.setText("10");
		fieldNy.setText("10");
		fieldNt.setText("1000");
		fieldTau.setText("0.001");
		fieldExprU.setText("cos(x)*cos(y)*e^(-2*a*t)");
		*/

		method.setA(Double.parseDouble(fieldA.getText()));
		method.setB(Double.parseDouble(fieldB.getText()));
		method.setExprF(new ExpTree(fieldExprF.getText()));
		method.setExprPsi(new ExpTree(fieldExprPsi.getText()));
		method.setAlpha1(Double.parseDouble(fieldAlpha1.getText()));
		method.setBeta1(Double.parseDouble(fieldBeta1.getText()));
		method.setAlpha2(Double.parseDouble(fieldAlpha2.getText()));
		method.setBeta2(Double.parseDouble(fieldBeta2.getText()));
		method.setAlpha3(Double.parseDouble(fieldAlpha3.getText()));
		method.setBeta3(Double.parseDouble(fieldBeta3.getText()));
		method.setAlpha4(Double.parseDouble(fieldAlpha4.getText()));
		method.setBeta4(Double.parseDouble(fieldBeta4.getText()));
		method.setExprFi1(new ExpTree(fieldExprFi1.getText()));
		method.setExprFi2(new ExpTree(fieldExprFi2.getText()));
		method.setExprFi3(new ExpTree(fieldExprFi3.getText()));
		method.setExprFi4(new ExpTree(fieldExprFi4.getText()));
		method.setLx(Double.parseDouble(fieldLx.getText()));
		method.setLy(Double.parseDouble(fieldLy.getText()));
		method.setNx(Integer.parseInt(fieldNx.getText()));
		method.setNy(Integer.parseInt(fieldNy.getText()));
		method.setNt(Integer.parseInt(fieldNt.getText()));
		method.setTau(Double.parseDouble(fieldTau.getText()));
		method.setExprU(new ExpTree(fieldExprU.getText()));

		method.solve(methodType, matU, vecX, vecY, vecT);

		for (int k = 0; k < matU.size(); ++k) {
			output.writeln(matU.get(k) + "\n");
		}

		output.close();

		rbSliceClick(null);
		updatePlotters();
	}

	public void rbSliceClick(ActionEvent actionEvent) {
		RadioButton rbSlice = (RadioButton)rbGroupSlice.getSelectedToggle();

		if (rbSlice == radioButtonSliceXY) {
			plotterSolutionAxisX.setLabel("t");
			plotterErrorAxisX.setLabel("t");
			scrollBarK1.setMax(Double.parseDouble(fieldNx.getText()));
			scrollBarK2.setMax(Double.parseDouble(fieldNy.getText()));
			sliceType = SLICE_XY;
		} else if (rbSlice == radioButtonSliceXT) {
			plotterSolutionAxisX.setLabel("y");
			plotterErrorAxisX.setLabel("y");
			scrollBarK1.setMax(Double.parseDouble(fieldNx.getText()));
			scrollBarK2.setMax(Double.parseDouble(fieldNt.getText()));
			sliceType = SLICE_XT;
		} else {
			plotterSolutionAxisX.setLabel("x");
			plotterErrorAxisX.setLabel("x");
			scrollBarK1.setMax(Double.parseDouble(fieldNy.getText()));
			scrollBarK2.setMax(Double.parseDouble(fieldNt.getText()));
			sliceType = SLICE_YT;
		}

		initPlotters();

		scrollBarK1.setValue(0.0);
		scrollBarK2.setValue(0.0);

		updatePlotters();
	}

	private void initPlotters() {
		seriesAnalytical = new XYChart.Series<>();
		seriesNumerical = new XYChart.Series<>();
		seriesError = new XYChart.Series<>();
		seriesAnalytical.setName("Анал.");
		seriesNumerical.setName("Числ.");

		if (sliceType == SLICE_XY) {
			seriesError.setName("e(t)");

			for (int j = 0; j < vecT.getSize(); ++j) {
				seriesAnalytical.getData().add(new XYChart.Data<>(0.0, 0.0));
				seriesNumerical.getData().add(new XYChart.Data<>(0.0, 0.0));
			}

			for (int k = 0; k < vecT.getSize(); ++k) {
				double error = 0.0;

				for (int i = 0; i < vecY.getSize(); ++i) {
					for (int j = 0; j < vecX.getSize(); ++j) {
						error = Math.max(error, Math.abs(matU.get(k).get(i, j) - method.u(vecX.get(j), vecY.get(i), vecT.get(k))));
					}
				}

				seriesError.getData().add(new XYChart.Data<>(vecT.get(k), error));
			}
		} else if (sliceType == SLICE_XT) {
			seriesError.setName("e(y)");

			for (int j = 0; j < vecY.getSize(); ++j) {
				seriesAnalytical.getData().add(new XYChart.Data<>(0.0, 0.0));
				seriesNumerical.getData().add(new XYChart.Data<>(0.0, 0.0));
			}

			for (int i = 0; i < vecY.getSize(); ++i) {
				double error = 0.0;

				for (int k = 0; k < vecT.getSize(); ++k) {
					for (int j = 0; j < vecX.getSize(); ++j) {
						error = Math.max(error, Math.abs(matU.get(k).get(i, j) - method.u(vecX.get(j), vecY.get(i), vecT.get(k))));
					}
				}

				seriesError.getData().add(new XYChart.Data<>(vecY.get(i), error));
			}
		} else {
			seriesError.setName("e(x)");

			for (int j = 0; j < vecX.getSize(); ++j) {
				seriesAnalytical.getData().add(new XYChart.Data<>(0.0, 0.0));
				seriesNumerical.getData().add(new XYChart.Data<>(0.0, 0.0));
			}

			for (int j = 0; j < vecX.getSize(); ++j) {
				double error = 0.0;

				for (int i = 0; i < vecY.getSize(); ++i) {
					for (int k = 0; k < vecT.getSize(); ++k) {
						error = Math.max(error, Math.abs(matU.get(k).get(i, j) - method.u(vecX.get(j), vecY.get(i), vecT.get(k))));
					}
				}

				seriesError.getData().add(new XYChart.Data<>(vecX.get(j), error));
			}
		}

		plotterSolution.setCreateSymbols(false);
		plotterSolution.getData().clear();
		plotterSolution.getData().add(seriesAnalytical);
		plotterSolution.getData().add(seriesNumerical);

		plotterError.setCreateSymbols(false);
		plotterError.getData().clear();
		plotterError.getData().add(seriesError);
	}

	private void updatePlotters() {
		int k1 = (int)scrollBarK1.getValue();
		int k2 = (int)scrollBarK2.getValue();

		labelK.setText("Параметры графика (k1 = " + k1 + ", k2 = " + k2 + ")");

		if (sliceType == SLICE_XY) {
			for (int k = 0; k < vecT.getSize(); ++k) {
				seriesAnalytical.getData().set(k, new XYChart.Data<>(vecT.get(k), method.u(vecX.get(k1), vecY.get(k2), vecT.get(k))));
				seriesNumerical.getData().set(k, new XYChart.Data<>(vecT.get(k), matU.get(k).get(k1, k2)));
			}
		} else if (sliceType == SLICE_XT) {
			for (int i = 0; i < vecY.getSize(); ++i) {
				seriesAnalytical.getData().set(i, new XYChart.Data<>(vecY.get(i), method.u(vecX.get(k1), vecY.get(i), vecT.get(k2))));
				seriesNumerical.getData().set(i, new XYChart.Data<>(vecY.get(i), matU.get(k2).get(i, k1)));
			}
		} else {
			for (int j = 0; j < vecX.getSize(); ++j) {
				seriesAnalytical.getData().set(j, new XYChart.Data<>(vecX.get(j), method.u(vecX.get(j), vecY.get(k1), vecT.get(k2))));
				seriesNumerical.getData().set(j, new XYChart.Data<>(vecX.get(j), matU.get(k2).get(k1, j)));
			}
		}
	}

	private int sliceType;
	private Parabolic2D method;
	private ArrayList<Matrix> matU;
	private Vector vecX;
	private Vector vecY;
	private Vector vecT;
	private XYChart.Series<Double, Double> seriesAnalytical;
	private XYChart.Series<Double, Double> seriesNumerical;
	private XYChart.Series<Double, Double> seriesError;

	private static final int SLICE_XY = 0;
	private static final int SLICE_XT = 1;
	private static final int SLICE_YT = 2;

	@FXML
	private LineChart<Double, Double> plotterSolution;
	@FXML
	private LineChart<Double, Double> plotterError;
	@FXML
	private NumberAxis plotterSolutionAxisX;
	@FXML
	private NumberAxis plotterErrorAxisX;
	@FXML
	private TextField fieldA;
	@FXML
	private TextField fieldB;
	@FXML
	private TextField fieldExprF;
	@FXML
	private TextField fieldAlpha1;
	@FXML
	private TextField fieldBeta1;
	@FXML
	private TextField fieldAlpha2;
	@FXML
	private TextField fieldBeta2;
	@FXML
	private TextField fieldAlpha3;
	@FXML
	private TextField fieldBeta3;
	@FXML
	private TextField fieldAlpha4;
	@FXML
	private TextField fieldBeta4;
	@FXML
	private TextField fieldExprFi1;
	@FXML
	private TextField fieldExprFi2;
	@FXML
	private TextField fieldExprFi3;
	@FXML
	private TextField fieldExprFi4;
	@FXML
	private TextField fieldLx;
	@FXML
	private TextField fieldLy;
	@FXML
	private TextField fieldExprPsi;
	@FXML
	private TextField fieldNx;
	@FXML
	private TextField fieldNy;
	@FXML
	private TextField fieldNt;
	@FXML
	private TextField fieldTau;
	@FXML
	private TextField fieldExprU;
	@FXML
	private ToggleGroup rbGroupMethod;
	@FXML
	private ToggleGroup rbGroupSlice;
	@FXML
	private RadioButton radioButtonADI;
	@FXML
	private RadioButton radioButtonSliceXY;
	@FXML
	private RadioButton radioButtonSliceXT;
	@FXML
	private Label labelK;
	@FXML
	private ScrollBar scrollBarK1;
	@FXML
	private ScrollBar scrollBarK2;
}
