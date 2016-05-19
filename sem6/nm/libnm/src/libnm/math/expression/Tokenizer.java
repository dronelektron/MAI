package libnm.math.expression;

import java.util.ArrayList;
import java.util.Stack;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class Tokenizer {
	Tokenizer() {
		m_patterns = new ArrayList<>();
		m_patterns.add(Pattern.compile("^(arcsin|arccos|arctg|arcctg|sin|cos|tg|ctg|sqrt|abs|ln)"));
		m_patterns.add(Pattern.compile("^[a-z]+[0-9]*"));
		m_patterns.add(Pattern.compile("^[0-9]+(\\.[0-9]+)?"));
		m_patterns.add(Pattern.compile("^[\\+\\-]"));
		m_patterns.add(Pattern.compile("^[*/]"));
		m_patterns.add(Pattern.compile("^[\\^]"));
		m_patterns.add(Pattern.compile("^[\\(]"));
		m_patterns.add(Pattern.compile("^[\\)]"));
	}

	Stack<ExpTreeNode> getTokens(String str) {
		int pos = 0;
		Stack<ExpTreeNode> stackOp = new Stack<>();
		Stack<ExpTreeNode> stackRes = new Stack<>();
		ExpTreeNode prevToken = null;

		while (true) {
			ExpTreeNode token = m_getToken(str, pos);

			if (token == null) {
				if (pos < str.length()) {
					System.out.println("Syntax error in column: " + pos);

					return null;
				}

				break;
			}

			if (token.getValue().equals("-") && (prevToken == null || isOpenBracket(prevToken))) {
				token.setSign(true);
			}

			if (isTerm(token)) {
				stackRes.push(token);
			} else {
				while (!stackOp.empty()) {
					ExpTreeNode topNode = stackOp.peek();

					if (m_isHigher(topNode, token) && !isOpenBracket(topNode)) {
						stackRes.push(topNode);
						stackOp.pop();
					} else {
						break;
					}
				}

				stackOp.push(token);

				if (isCloseBracket(stackOp.peek())) {
					stackOp.pop();
					stackOp.pop();
				}
			}

			pos += token.getValue().length();
			prevToken = token;
		}

		while (!stackOp.empty()) {
			stackRes.push(stackOp.peek());
			stackOp.pop();
		}

		return stackRes;
	}

	static boolean isFunc(ExpTreeNode root) {
		return root.getType() == FUNCTION;
	}

	static boolean isVar(ExpTreeNode root) {
		return root.getType() == VARIABLE;
	}

	static boolean isVal(ExpTreeNode root) {
		return root.getType() == VALUE;
	}

	static boolean isPlusMinus(ExpTreeNode root) {
		return root.getType() == PLUSMINUS;
	}

	static boolean isMulDiv(ExpTreeNode root) {
		return root.getType() == MULDIV;
	}

	static boolean isPower(ExpTreeNode root) {
		return root.getType() == POWER;
	}

	static boolean isOpenBracket(ExpTreeNode root) {
		return root.getType() == OPEN_BRACKET;
	}

	static boolean isCloseBracket(ExpTreeNode root) {
		return root.getType() == CLOSE_BRACKET;
	}

	static boolean isTerm(ExpTreeNode root) {
		return isVar(root) || isVal(root);
	}

	static boolean isOp(ExpTreeNode root) {
		return isPlusMinus(root) || isMulDiv(root) || isPower(root);
	}

	private ExpTreeNode m_getToken(String str, int start) {
		for (int i = 0; i < m_patterns.size(); ++i) {
			Matcher m = m_patterns.get(i).matcher(str.substring(start));

			if (m.find()) {
				return new ExpTreeNode(m.group(), i, false);
			}
		}

		return null;
	}

	private boolean m_isHigher(ExpTreeNode node1, ExpTreeNode node2) {
		if (isPower(node1) && isPower(node2)) {
			return false;
		} else if (isOpenBracket(node2)) {
			return false;
		}

		return m_priority(node1) >= m_priority(node2);
	}

	private int m_priority(ExpTreeNode node) {
		if (isFunc(node)) {
			return 4;
		} else if (isPower(node)) {
			return 3;
		} else if (isMulDiv(node)) {
			return 2;
		} else if (isPlusMinus(node)) {
			return 1;
		}

		return 0;
	}

	static final int FUNCTION = 0;
	static final int VARIABLE = 1;
	static final int VALUE = 2;
	static final int PLUSMINUS = 3;
	static final int MULDIV = 4;
	static final int POWER = 5;
	static final int OPEN_BRACKET = 6;
	static final int CLOSE_BRACKET = 7;

	private ArrayList<Pattern> m_patterns;
}
