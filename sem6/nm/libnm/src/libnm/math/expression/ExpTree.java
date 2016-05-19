package libnm.math.expression;

import java.util.HashMap;
import java.util.Stack;

public class ExpTree {
	public ExpTree(ExpTree tree) {
		m_root = m_copy(tree.m_root);
		m_vars = new HashMap<>(tree.m_vars);
	}

	public ExpTree(String str) {
		Tokenizer tokenizer = new Tokenizer();
		Stack<ExpTreeNode> tokens = tokenizer.getTokens(str);

		m_vars = new HashMap<>();
		m_vars.put("pi", Math.PI);
		m_vars.put("e", Math.E);

		m_root = tokens.peek();

		m_buildTree(tokens);
	}

	public ExpTree setVar(String var, double value) {
		m_vars.put(var, value);

		return this;
	}

	public double calculate() {
		return m_calculate(m_root);
	}

	public ExpTree derivateBy(String var) {
		ExpTree res = new ExpTree(this);

		m_derivateBy(res.m_root, var);

		return res;
	}
	/*
	public void printTree() {
		System.out.println("--------");
		m_printTree(m_root, 0);
		System.out.println("--------");
	}
	*/
	public String getExpr() {
		StringBuilder res = new StringBuilder();

		m_printExpr(m_root, res);

		return res.toString();
	}

	private void m_printExpr(ExpTreeNode root, StringBuilder sb) {
		if (root == null) {
			return;
		}

		sb.append("(");
		m_printExpr(root.getLeft(), sb);
		sb.append(root.getValue());
		m_printExpr(root.getRight(), sb);
		sb.append(")");
	}

	private void m_buildTree(Stack<ExpTreeNode> stack) {
		if (stack.empty()) {
			return;
		}

		ExpTreeNode root = stack.peek();
		stack.pop();

		if (Tokenizer.isTerm(root)) {
			return;
		}

		root.setRight(stack.peek());
		m_buildTree(stack);

		if (!Tokenizer.isFunc(root) && !root.getSign()) {
			root.setLeft(stack.peek());
			m_buildTree(stack);
		}
	}
	/*
	private void m_printTree(ExpTreeNode root, int level) {
		if (root != null) {
			m_printTree(root.getRight(), level + 1);

			for (int i = 0; i < level; ++i) {
				System.out.print("\t");
			}

			System.out.println(root.getValue());

			m_printTree(root.getLeft(), level + 1);
		}
	}
	*/
	private double m_calculate(ExpTreeNode root) {
		if (Tokenizer.isVar(root) && m_vars.containsKey(root.getValue())) {
			return m_vars.get(root.getValue());
		} else if (Tokenizer.isVal(root)) {
			return Double.parseDouble(root.getValue());
		} else if (Tokenizer.isFunc(root)) {
			double val = m_calculate(root.getRight());

			switch (root.getValue()) {
				case "arcsin":
					return Math.asin(val);

				case "arccos":
					return Math.acos(val);

				case "arctg":
					return Math.atan(val);

				case "arcctg":
					return Math.atan(-val) + Math.PI * 0.5;

				case "sin":
					return Math.sin(val);

				case "cos":
					return Math.cos(val);

				case "tg":
					return Math.tan(val);

				case "ctg":
					return 1.0 / Math.tan(val);

				case "sqrt":
					return Math.sqrt(val);

				case "abs":
					return Math.abs(val);

				case "ln":
					return Math.log(val);
			}
		} else if (Tokenizer.isOp(root)) {
			double leftVal = root.getSign() ? 0.0 : m_calculate(root.getLeft());
			double rightVal = m_calculate(root.getRight());

			switch (root.getValue()) {
				case "+":
					return leftVal + rightVal;

				case "-":
					return leftVal - rightVal;

				case "*":
					return leftVal * rightVal;

				case "/":
					return leftVal / rightVal;

				case "^":
					return Math.pow(leftVal, rightVal);
			}
		}

		return 0.0;
	}

	private boolean isConstant(ExpTreeNode root) {
		if (root == null || Tokenizer.isVal(root)) {
			return true;
		} else if (Tokenizer.isOp(root)) {
			return isConstant(root.getLeft()) && isConstant(root.getRight());
		}

		return false;
	}

	private ExpTreeNode m_copyNode(ExpTreeNode root) {
		return new ExpTreeNode(root.getValue(), root.getType(), root.getSign());
	}

	private ExpTreeNode m_copy(ExpTreeNode root) {
		ExpTreeNode node = m_copyNode(root);

		if (root.getLeft() != null) {
			node.setLeft(m_copy(root.getLeft()));
		}

		if (root.getRight() != null) {
			node.setRight(m_copy(root.getRight()));
		}

		return node;
	}

	private void m_derivateBy(ExpTreeNode root, String var) {
		if (Tokenizer.isPlusMinus(root)) {
			m_derivateBy(root.getRight(), var);

			if (!Tokenizer.isFunc(root) && !root.getSign()) {
				m_derivateBy(root.getLeft(), var);
			}
		} else if (Tokenizer.isMulDiv(root)) {
			if (root.getValue().equals("*")) {
				ExpTreeNode left = m_copy(root);
				ExpTreeNode right = m_copy(root);

				root.setValue("+");
				root.setType(Tokenizer.PLUSMINUS);
				root.setLeft(left);
				root.setRight(right);

				m_derivateBy(left.getLeft(), var);
				m_derivateBy(right.getRight(), var);
			} else {
				ExpTreeNode left = new ExpTreeNode("-", Tokenizer.PLUSMINUS, false);
				ExpTreeNode leftLeft = new ExpTreeNode("*", Tokenizer.MULDIV, false);
				ExpTreeNode right = new ExpTreeNode("^", Tokenizer.POWER, false);

				leftLeft.setLeft(m_copy(root.getLeft()));
				leftLeft.setRight(m_copy(root.getRight()));
				left.setLeft(leftLeft);
				left.setRight(m_copy(leftLeft));
				right.setLeft(m_copy(root.getRight()));
				right.setRight(new ExpTreeNode("2.0", Tokenizer.VALUE, false));
				root.setLeft(left);
				root.setRight(right);

				m_derivateBy(leftLeft.getLeft(), var);
				m_derivateBy(left.getRight().getRight(), var);
			}
		} else if (Tokenizer.isPower(root)) {
			if (root.getLeft().getValue().equals("e")) {
				root.setLeft(m_copy(root));
			} else if (isConstant(root.getLeft())) {
				ExpTreeNode node = m_copy(root);

				root.setLeft(new ExpTreeNode("*", Tokenizer.MULDIV, false));
				root.getLeft().setLeft(node);
				root.getLeft().setRight(new ExpTreeNode("ln", Tokenizer.FUNCTION, false));
				root.getLeft().getRight().setRight(m_copy(node.getLeft()));
			} else {
				ExpTreeNode left = root.getLeft();

				root.setLeft(new ExpTreeNode("*", Tokenizer.MULDIV, false));
				root.getLeft().setLeft(m_copy(root.getRight()));
				root.getLeft().setRight(new ExpTreeNode("^", Tokenizer.POWER, false));
				root.getLeft().getRight().setLeft(left);
				root.getLeft().getRight().setRight(new ExpTreeNode("-", Tokenizer.PLUSMINUS, false));
				root.getLeft().getRight().getRight().setLeft(m_copy(root.getRight()));
				root.getLeft().getRight().getRight().setRight(new ExpTreeNode("1.0", Tokenizer.VALUE, false));
				root.setRight(m_copy(left));
			}

			root.setValue("*");
			root.setType(Tokenizer.MULDIV);

			m_derivateBy(root.getRight(), var);
		} else if (Tokenizer.isVar(root)) {
			if (root.getValue().equals(var)) {
				root.setValue("1.0");
			} else {
				root.setValue("0.0");
			}

			root.setType(Tokenizer.VALUE);
		} else if (Tokenizer.isVal(root)) {
			root.setValue("0.0");
		} else if (Tokenizer.isFunc(root)) {
			switch (root.getValue()) {
				case "sin":
					root.setLeft(new ExpTreeNode("cos", Tokenizer.FUNCTION, false));
					root.getLeft().setRight(m_copy(root.getRight()));

					break;

				case "cos":
					root.setLeft(new ExpTreeNode("-", Tokenizer.PLUSMINUS, true));
					root.getLeft().setRight(new ExpTreeNode("sin", Tokenizer.FUNCTION, false));
					root.getLeft().getRight().setRight(m_copy(root.getRight()));

					break;

				case "tg":
					root.setLeft(new ExpTreeNode("/", Tokenizer.MULDIV, false));
					root.getLeft().setLeft(new ExpTreeNode("1.0", Tokenizer.VALUE, false));
					root.getLeft().setRight(new ExpTreeNode("^", Tokenizer.POWER, false));
					root.getLeft().getRight().setLeft(new ExpTreeNode("cos", Tokenizer.FUNCTION, false));
					root.getLeft().getRight().getLeft().setRight(m_copy(root.getRight()));
					root.getLeft().getRight().setRight(new ExpTreeNode("2.0", Tokenizer.VALUE, false));

					break;

				case "ctg":
					root.setLeft(new ExpTreeNode("/", Tokenizer.MULDIV, false));
					root.getLeft().setLeft(new ExpTreeNode("-", Tokenizer.PLUSMINUS, true));
					root.getLeft().getLeft().setRight(new ExpTreeNode("1.0", Tokenizer.VALUE, false));
					root.getLeft().setRight(new ExpTreeNode("^", Tokenizer.POWER, false));
					root.getLeft().getRight().setLeft(new ExpTreeNode("sin", Tokenizer.POWER, false));
					root.getLeft().getRight().getLeft().setRight(m_copy(root.getRight()));
					root.getLeft().getRight().setRight(new ExpTreeNode("2.0", Tokenizer.VALUE, false));

					break;

				case "sqrt":
					ExpTreeNode node = m_copy(root);

					root.setLeft(new ExpTreeNode("/", Tokenizer.MULDIV, false));
					root.getLeft().setLeft(new ExpTreeNode("1.0", Tokenizer.VALUE, false));
					root.getLeft().setRight(new ExpTreeNode("*", Tokenizer.MULDIV, false));
					root.getLeft().getRight().setLeft(new ExpTreeNode("2.0", Tokenizer.VALUE, false));
					root.getLeft().getRight().setRight(node);

					break;

				case "abs":
					root.setLeft(new ExpTreeNode("/", Tokenizer.MULDIV, false));
					root.getLeft().setLeft(m_copy(root.getRight()));
					root.getLeft().setRight(new ExpTreeNode("abs", Tokenizer.FUNCTION, false));
					root.getLeft().getRight().setRight(m_copy(root.getRight()));

					break;

				case "ln":
					root.setLeft(new ExpTreeNode("/", Tokenizer.MULDIV, false));
					root.getLeft().setLeft(new ExpTreeNode("1.0", Tokenizer.VALUE, false));
					root.getLeft().setRight(m_copy(root.getRight()));

					break;
			}

			root.setValue("*");
			root.setType(Tokenizer.MULDIV);

			m_derivateBy(root.getRight(), var);
		}
	}

	private ExpTreeNode m_root;
	private HashMap<String, Double> m_vars;
}
