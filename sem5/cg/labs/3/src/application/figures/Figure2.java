package application.figures;

import javafx.scene.control.Slider;
import math.Vector;

public class Figure2 extends Figure {
	public Figure2(Slider[] params) {
		super();

		this.params = params;
	}

	@Override
	public void generate() {
		double paramR = params[0].getValue();
		double paramr = params[1].getValue();
		int paramRStep = (int)params[2].getValue();
		int paramrStep = (int)params[3].getValue();
		double angleRStep = 2.0 * Math.PI / paramRStep;
		double anglerStep = 2.0 * Math.PI / paramrStep;

		points.clear();
		indexes.clear();

		for (int i = 0; i < paramRStep; ++i) {
			for (int j = 0; j < paramrStep; ++j) {
				double fi = j * anglerStep;
				double ksi = i * angleRStep;
				double x = (paramR + paramr * Math.cos(fi)) * Math.cos(ksi);
				double y = (paramR + paramr * Math.cos(fi)) * Math.sin(ksi);
				double z = paramr * Math.sin(fi);

				points.add(new Vector(x, y, z, 1.0));
			}
		}

		for (int i = 0; i < paramRStep; ++i) {
			for (int j = 0; j < paramrStep; ++j) {
				int offsetLeftUp = i * paramrStep + j;
				int offsetLeftDown = i * paramrStep + (j + 1) % paramrStep;
				int offsetRightUp = (offsetLeftUp + paramrStep) % points.size();
				int offsetRightDown = (offsetLeftDown + paramrStep) % points.size();

				indexes.add(offsetLeftDown);
				indexes.add(offsetLeftUp);
				indexes.add(offsetRightDown);
				indexes.add(offsetRightDown);
				indexes.add(offsetLeftUp);
				indexes.add(offsetRightUp);
			}
		}
	}
}
