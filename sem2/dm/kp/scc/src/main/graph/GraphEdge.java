package main.graph;

public class GraphEdge {
	public GraphEdge(GraphVertex start, GraphVertex end) {
		this.start = start;
		this.end = end;
	}

	public void setStart(GraphVertex start) {
		this.start = start;
	}

	public void setEnd(GraphVertex end) {
		this.end = end;
	}

	public GraphVertex getStart() {
		return start;
	}

	public GraphVertex getEnd() {
		return end;
	}

	private GraphVertex start;
	private GraphVertex end;
}
