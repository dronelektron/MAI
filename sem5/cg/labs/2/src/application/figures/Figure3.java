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
		double paramHeight = params[0].getValue();
		double paramRadius = params[1].getValue();
		int paramSides = (int)params[2].getValue();
		double paramAngle = params[3].getValue();
		double step = paramAngle * Math.PI / (180.0 * paramSides);

		points.clear();
		indexes.clear();

		for (int i = 0; i < paramSides; ++i) {
			double x = Math.cos(i * step) * paramRadius;
			double z = Math.sin(i * step) * paramRadius;

			points.add(new Vector(x, 0.0, z, 1.0));
		}

		points.add(new Vector(0.0, 0.0, 0.0, 1.0));
		points.add(new Vector(0.0, paramHeight, 0.0, 1.0));

		for (int i = 0; i < paramSides - 1; ++i) {
			indexes.add(points.size() - 2);
			indexes.add(i);
			indexes.add(i + 1);
		}

		for (int i = 0; i < paramSides - 1; ++i) {
			indexes.add(points.size() - 1);
			indexes.add(i + 1);
			indexes.add(i);
		}

		indexes.add(0);
		indexes.add(points.size() - 2);
		indexes.add(points.size() - 1);
		indexes.add(points.size() - 1);
		indexes.add(points.size() - 2);
		indexes.add(points.size() - 3);
	}
}
