package main.graph;

import java.util.ArrayList;
import java.util.HashMap;

public class Graph {
	public Graph() {
		vertices = new ArrayList<>();
		edges = new ArrayList<>();
		ids = new HashMap<>();
	}

	public void addVertex(double x, double y) {
		if (findVertex(x, y) == null) {
			GraphVertex gv = new GraphVertex(x, y);

			vertices.add(gv);
			ids.put(gv, vertices.size());
		}
	}

	public void removeVertex(GraphVertex vertex) {
		ArrayList<GraphEdge> edgesNew = new ArrayList<>();

		for (GraphEdge ge : edges) {
			if (ge.getStart() != vertex && ge.getEnd() != vertex) {
				edgesNew.add(ge);
			}
		}

		edges = edgesNew;
		vertices.remove(vertex);
		ids.clear();

		for (int i = 0; i < vertices.size(); ++i) {
			ids.put(vertices.get(i), i + 1);
		}
	}

	public void addEdge(GraphVertex start, GraphVertex end) {
		if (start != end && findEdge(start, end) == null) {
			edges.add(new GraphEdge(start, end));
		}
	}

	public void removeEdge(GraphEdge edge) {
		edges.remove(edge);
	}

	public void reverseEdges() {
		for (GraphEdge ge : edges) {
			GraphVertex tmp = ge.getStart();

			ge.setStart(ge.getEnd());
			ge.setEnd(tmp);
		}
	}

	public void clear() {
		vertices.clear();
		edges.clear();
		ids.clear();
	}

	public int getVertexId(GraphVertex vertex) {
		return ids.get(vertex);
	}

	public GraphVertex findVertex(double x, double y) {
		for (GraphVertex gv : vertices) {
			double vx = gv.getX();
			double vy = gv.getY();
			double dx = vx - x;
			double dy = vy - y;
			double r = GraphVertex.RADIUS;

			if (dx * dx + dy * dy <= r * r) {
				return gv;
			}
		}

		return null;
	}

	public GraphEdge findEdge(double x1, double y1, double x2, double y2) {
		for (GraphEdge ge : edges) {
			GraphVertex start = ge.getStart();
			GraphVertex end = ge.getEnd();

			if (isCross(x1, y1, x2, y2, start.getX(), start.getY(), end.getX(), end.getY())) {
				return ge;
			}
		}

		return null;
	}

	public GraphEdge findEdge(GraphVertex start, GraphVertex end) {
		for (GraphEdge ge : edges) {
			if (ge.getStart() == start && ge.getEnd() == end) {
				return ge;
			}
		}

		return null;
	}

	public ArrayList<GraphVertex> getChildVertices(GraphVertex vertex) {
		ArrayList<GraphVertex> res = new ArrayList<>();

		for (GraphEdge ge : edges) {
			if (ge.getStart() == vertex) {
				res.add(ge.getEnd());
			}
		}

		return res;
	}

	public ArrayList<GraphVertex> getVertices() {
		return vertices;
	}

	public ArrayList<GraphEdge> getEdges() {
		return edges;
	}

	private double linePointZone(double x1, double y1, double x2, double y2, double x, double y) {
		return (y1 - y2) * x + (x2 - x1) * y + (x1 * y2 - x2 * y1);
	}

	private boolean isCross(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4) {
		return linePointZone(x1, y1, x2, y2, x3, y3) * linePointZone(x1, y1, x2, y2, x4, y4) <= 0.0 &&
				linePointZone(x3, y3, x4, y4, x1, y1) * linePointZone(x3, y3, x4, y4, x2, y2) <= 0.0;
	}

	private ArrayList<GraphVertex> vertices;
	private ArrayList<GraphEdge> edges;
	private HashMap<GraphVertex, Integer> ids;
}
