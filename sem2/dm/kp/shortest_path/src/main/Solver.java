package main;

import javafx.scene.control.TextArea;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import main.graph.*;

public class Solver {
	public Solver(Graph graph, TextArea textArea) {
		this.graph = graph;
		this.textArea = textArea;
	}

	public void shortestPath() {
		final int INF = 1 << 30;
		ArrayList<GraphVertex> vertices = graph.getVertices();
		int n = vertices.size();

		textArea.clear();

		for (int k = 0; k < n; ++k) {
			HashMap<GraphVertex, Integer> dist = new HashMap<>();
			HashMap<GraphVertex, GraphVertex> from = new HashMap<>();
			HashSet<GraphVertex> used = new HashSet<>();

			for (int i = 0; i < n; ++i) {
				dist.put(vertices.get(i), INF);
				from.put(vertices.get(i), null);
			}

			dist.put(vertices.get(k), 0);

			textArea.appendText("Расстояние от вершины " + graph.getVertexId(vertices.get(k)) + ":\n");

			for (int i = 0; i < n; ++i) {
				int cur = -1;

				for (int j = 0; j < n; ++j) {
					if (!used.contains(vertices.get(j)) &&
							(cur == -1 || dist.get(vertices.get(j)) < dist.get(vertices.get(cur)))) {
						cur = j;
					}
				}

				if (dist.get(vertices.get(cur)) == INF) {
					break;
				}

				used.add(vertices.get(cur));

				ArrayList<GraphVertex> child = graph.getChildVertices(vertices.get(cur));

				for (int j = 0; j < child.size(); ++j) {
					int curDist = dist.get(child.get(j));
					int newDist = dist.get(vertices.get(cur)) + graph.findEdge(vertices.get(cur), child.get(j)).getCost();

					if (newDist < curDist) {
						dist.put(child.get(j), newDist);
						from.put(child.get(j), vertices.get(cur));
					}
				}
			}

			for (int i = 0; i < n; ++i) {
				GraphVertex gv = vertices.get(i);
				GraphVertex tmp = gv;
				ArrayList<GraphVertex> path = new ArrayList<>();

				while (tmp != null) {
					path.add(tmp);
					tmp = from.get(tmp);
				}

				int[] pathIds = new int[path.size()];

				for (int j = path.size() - 1; j >= 0; --j) {
					pathIds[path.size() - 1 - j] = graph.getVertexId(path.get(j));
				}

				if (dist.get(gv) != INF && i != k) {
					textArea.appendText("До вершины " + graph.getVertexId(gv) + ": " + dist.get(gv) + "\n");
					textArea.appendText("Путь: " + Arrays.toString(pathIds) + "\n");
				}
			}
		}
	}

	private Graph graph;
	private TextArea textArea;
}
