package libnm.math.expression;

class ExpTreeNode {
	ExpTreeNode(String value, int type, boolean sign) {
		m_sign = sign;
		m_type = type;
		m_value = value;
		m_left = null;
		m_right = null;
	}

	int getType() {
		return m_type;
	}

	void setType(int type) {
		m_type = type;
	}

	boolean getSign() {
		return m_sign;
	}

	void setSign(boolean sign) {
		m_sign = sign;
	}

	String getValue() {
		return m_value;
	}

	void setValue(String value) {
		m_value = value;
	}

	ExpTreeNode getLeft() {
		return m_left;
	}

	void setLeft(ExpTreeNode left) {
		m_left = left;
	}

	ExpTreeNode getRight() {
		return m_right;
	}

	void setRight(ExpTreeNode right) {
		m_right = right;
	}

	private int m_type;
	private boolean m_sign;
	private String m_value;
	private ExpTreeNode m_left;
	private ExpTreeNode m_right;
}
