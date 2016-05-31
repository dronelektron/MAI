package main.graph;

public class GraphEdge {
	public GraphEdge(GraphVertex start, GraphVertex end) {
		this.start = start;
		this.end = end;
		this.cost = 0;
		this.areaX = 0.0;
		this.areaY = 0.0;
	}

	public void setStart(GraphVertex start) {
		this.start = start;
	}

	public void setEnd(GraphVertex end) {
		this.end = end;
	}

	public void setCost(int cost) {
		this.cost = cost;
	}

	public void setAreaX(double areaX) {
		this.areaX = areaX;
	}

	public void setAreaY(double areaY) {
		this.areaY = areaY;
	}

	public GraphVertex getStart() {
		return start;
	}

	public GraphVertex getEnd() {
		return end;
	}

	public int getCost() {
		return cost;
	}

	public double getAreaX() {
		return areaX;
	}

	public double getAreaY() {
		return areaY;
	}

	private GraphVertex start;
	private GraphVertex end;
	private int cost;
	private double areaX;
	private double areaY;
}
