package application.figures;

import javafx.scene.control.Slider;
import math.Vector;

public class Figure1 extends Figure {
	public Figure1(Slider[] params) {
		super();

		this.params = params;
	}

	@Override
	public void generate() {
		double paramHeight = params[0].getValue();
		double paramRadius = params[1].getValue();
		int paramSides = (int)params[2].getValue();
		double step = 2.0 * Math.PI / paramSides;

		points.clear();
		indexes.clear();

		for (int i = 0; i < paramSides; ++i) {
			double x = Math.cos(i * step) * paramRadius;
			double z = Math.sin(i * step) * paramRadius;

			points.add(new Vector(x, 0.0, z, 1.0));
			points.add(new Vector(x, paramHeight, z, 1.0));
		}

		points.add(new Vector(0.0, 0.0, 0.0, 1.0));
		points.add(new Vector(0.0, paramHeight, 0.0, 1.0));

		for (int i = 0; i < paramSides; ++i) {
			int offsetPoint = i * 2;

			indexes.add(offsetPoint);
			indexes.add(offsetPoint + 1);
			indexes.add((offsetPoint + 2) % (paramSides * 2));
			indexes.add((offsetPoint + 2) % (paramSides * 2));
			indexes.add(offsetPoint + 1);
			indexes.add((offsetPoint + 3) % (paramSides * 2));
		}

		for (int i = 0; i < paramSides; ++i) {
			indexes.add(points.size() - 2);
			indexes.add(i * 2);
			indexes.add((i * 2 + 2) % (paramSides * 2));
		}

		for (int i = 0; i < paramSides; ++i) {
			indexes.add(points.size() - 1);
			indexes.add((i * 2 + 3) % (paramSides * 2));
			indexes.add(i * 2 + 1);
		}
	}
}
