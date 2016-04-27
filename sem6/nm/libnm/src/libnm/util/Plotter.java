package libnm.util;

import java.awt.*;
import java.io.FileOutputStream;
import java.io.IOException;
import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.*;
import de.erichseifert.gral.graphics.Label;
import de.erichseifert.gral.io.plots.DrawableWriter;
import de.erichseifert.gral.io.plots.DrawableWriterFactory;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.axes.AxisRenderer;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;
import libnm.math.Vector;

public class Plotter {
	public Plotter(double width, double height) {
		m_width = width;
		m_height = height;

		m_initPlot();
	}

	public void addData(Vector vecX, Vector vecY, Color color, String legend) {
		DataTable data = new DataTable(Double.class, Double.class);
		PointRenderer pointRenderer = new DefaultPointRenderer2D();
		LineRenderer lineRenderer = new DefaultLineRenderer2D();
		DataSeries dataSeries = new DataSeries(legend, data);

		for (int i = 0; i < vecX.getSize(); ++i) {
			data.add(vecX.get(i), vecY.get(i));
		}

		pointRenderer.setColor(color);
		lineRenderer.setColor(color);

		m_plot.add(dataSeries);
		m_plot.setPointRenderers(dataSeries, pointRenderer);
		m_plot.setLineRenderers(dataSeries, lineRenderer);
	}

	public void clearData() {
		m_initPlot();
	}

	public void savePng(String fileName) {
		DrawableWriter writer = DrawableWriterFactory.getInstance().get("image/png");

		try {
			writer.write(m_plot, new FileOutputStream(fileName), m_width, m_height);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void m_initPlot() {
		final double MARGIN = 64.0;

		m_plot = new XYPlot();
		m_plot.setInsets(new Insets2D.Double(MARGIN, MARGIN, MARGIN, MARGIN));
		m_plot.setLegendVisible(true);

		AxisRenderer axisX = m_plot.getAxisRenderer(XYPlot.AXIS_X);
		AxisRenderer axisY = m_plot.getAxisRenderer(XYPlot.AXIS_Y);

		axisX.setLabel(new Label("X"));
		axisY.setLabel(new Label("Y"));
	}

	private double m_width;
	private double m_height;
	private XYPlot m_plot;
}
