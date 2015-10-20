package application.figures;

import javafx.scene.control.Slider;
import math.Vector;

public class Figure3 extends Figure {
	public Figure3(Slider[] params) {
		super();

		this.params = params;
	}

	@Override
	public void generate() {
		double paramRadius = params[0].getValue();
		double paramAngle = params[1].getValue();
		int paramStepV = (int)params[2].getValue();
		int paramStepH = (int)params[3].getValue();
		double deltaAngleV = paramAngle * Math.PI / (180.0 * paramStepV);
		double deltaAngleH = 2.0 * Math.PI / paramStepH;

		points.clear();
		indexes.clear();

		for (int i = 0; i < paramStepH; ++i) {
			double curAngleH = i * deltaAngleH;

			for (int j = 0; j < paramStepV; ++j) {
				double curAngleV = (j + 1) * deltaAngleV;
				double x = paramRadius * Math.sin(curAngleV) * Math.cos(curAngleH);
				double z = paramRadius * Math.sin(curAngleV) * Math.sin(curAngleH);
				double y = paramRadius * Math.cos(curAngleV);

				points.add(new Vector(x, y, z, 1.0));
			}
		}

		points.add(new Vector(0.0, 0.0, 0.0, 1.0));
		points.add(new Vector(0.0, paramRadius, 0.0, 1.0));

		int mod = points.size() - 2;

		for (int i = 0; i < paramStepH; ++i) {
			for (int j = 0; j < paramStepV - 1; ++j) {
				int offset = i * paramStepV + j;

				indexes.add(offset + 1);
				indexes.add(offset);
				indexes.add((offset + paramStepV + 1) % mod);
				indexes.add((offset + paramStepV + 1) % mod);
				indexes.add(offset);
				indexes.add((offset + paramStepV) % mod);
			}
		}

		for (int i = 0; i < paramStepH; ++i) {
			int offset = i * paramStepV;

			indexes.add(points.size() - 1);
			indexes.add((offset + paramStepV) % mod);
			indexes.add(offset);
		}

		for (int i = 0; i < paramStepH; ++i) {
			int offset = (i + 1) * paramStepV - 1;

			indexes.add(points.size() - 2);
			indexes.add(offset);
			indexes.add((offset + paramStepV) % mod);
		}
	}
}
