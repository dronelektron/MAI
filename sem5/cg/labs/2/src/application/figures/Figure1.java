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
		double paramRadius = params[0].getValue();
		int paramStepVert = (int)params[1].getValue();
		int paramStepHor = (int)params[2].getValue();
		double angleStepVert = Math.PI / paramStepVert;
		double angleStepHor = 2.0 * Math.PI / paramStepHor;

		points.clear();
		indexes.clear();

		for (int i = 0; i < paramStepHor; ++i)
		{
			for (int j = 1; j < paramStepVert; ++j)
			{
				double angleHor = i * angleStepHor;
				double angleVert = j * angleStepVert;
				double x = paramRadius * Math.cos(angleHor) * Math.sin(angleVert);
				double z = paramRadius * Math.sin(angleHor) * Math.sin(angleVert);
				double y = paramRadius * Math.cos(angleVert);

				points.add(new Vector(x, y, z, 1.0));
			}
		}

		points.add(new Vector(0.0, -paramRadius, 0.0, 1.0));
		points.add(new Vector(0.0, paramRadius, 0.0, 1.0));

		int mod = points.size() - 2;

		for (int i = 0; i < paramStepHor; ++i)
		{
			for (int j = 0; j < paramStepVert - 2; ++j)
			{
				int offset = i * (paramStepVert - 1) + j;

				indexes.add(offset + 1);
				indexes.add(offset);
				indexes.add((offset + paramStepVert) % mod);
				indexes.add((offset + paramStepVert) % mod);
				indexes.add(offset);
				indexes.add((offset + paramStepVert - 1) % mod);
			}
		}

		for (int i = 0; i < paramStepHor; ++i)
		{
			int offset = i * (paramStepVert - 1) + paramStepVert - 2;

			indexes.add(points.size() - 2);
			indexes.add(offset);
			indexes.add((offset + paramStepVert - 1) % mod);
		}

		for (int i = 0; i < paramStepHor; ++i)
		{
			int offset = i * (paramStepVert - 1);

			indexes.add(points.size() - 1);
			indexes.add((offset + paramStepVert - 1) % mod);
			indexes.add(offset);
		}
	}
}
