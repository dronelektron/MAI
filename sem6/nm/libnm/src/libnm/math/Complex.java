package libnm.math;

public class Complex {
	public Complex(double re, double im) {
		m_re = re;
		m_im = im;
	}

	public double getRe() {
		return m_re;
	}

	public void setRe(double re) {
		m_re = re;
	}

	public double getIm() {
		return m_im;
	}

	public void setIm(double im) {
		m_im = im;
	}

	public Complex sub(Complex other) {
		return new Complex(m_re - other.getRe(), m_im - other.getIm());
	}

	public double abs() {
		return Math.sqrt(Math.pow(m_re, 2.0) + Math.pow(m_im, 2.0));
	}

	@Override
	public String toString() {
		String res = String.valueOf(m_re);

		if (m_im != 0.0) {
			if (m_im > 0.0) {
				res += "+";
			}

			res += m_im + "*i";
		}

		return res;
	}

	private double m_re;
	private double m_im;
}
