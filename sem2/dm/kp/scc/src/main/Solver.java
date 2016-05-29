package main;

import java.util.ArrayList;
import java.util.Stack;
import javafx.scene.control.TextArea;

import main.graph.*;

public class Solver {
	public Solver(Graph graph, TextArea textArea) {
		this.graph = graph;
		this.textArea = textArea;
	}

	public void scc() {
		ArrayList<GraphVertex> vertices = graph.getVertices();

		stack = new Stack<>();
		used = new ArrayList<>();
		textArea.clear();

		for (GraphVertex gv : vertices) {
			if (!used.contains(gv)) {
				fillOrder(gv);
			}
		}

		graph.reverseEdges();
		used = new ArrayList<>();

		while (!stack.empty()) {
			GraphVertex gv = stack.peek();

			stack.pop();

			if (!used.contains(gv)) {
				dfs(gv);

				textArea.appendText("\n");
			}
		}

		graph.reverseEdges();
	}

	private void fillOrder(GraphVertex vertex) {
		ArrayList<GraphVertex> vertices = graph.getChildVertices(vertex);

		used.add(vertex);

		for (GraphVertex gv : vertices) {
			if (!used.contains(gv)) {
				fillOrder(gv);
			}
		}

		stack.push(vertex);
	}

	private void dfs(GraphVertex vertex) {
		ArrayList<GraphVertex> vertices = graph.getChildVertices(vertex);

		used.add(vertex);
		textArea.appendText(graph.getVertexId(vertex) + " ");

		for (GraphVertex gv : vertices) {
			if (!used.contains(gv)) {
				dfs(gv);
			}
		}
	}

	private Stack<GraphVertex> stack;
	private ArrayList<GraphVertex> used;
	private Graph graph;
	private TextArea textArea;
}
