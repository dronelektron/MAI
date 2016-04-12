package libnm.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import libnm.math.*;

public class Reader {
	public Reader(String fileName) {
		try {
			m_br = new BufferedReader(new FileReader(fileName));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public String readLine() {
		try {
			return m_br.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return null;
	}

	public int readInt() {
		try {
			return Integer.parseInt(m_br.readLine());
		} catch (IOException e) {
			e.printStackTrace();
		}

		return 0;
	}

	public double readDouble() {
		try {
			return Double.parseDouble(m_br.readLine());
		} catch (IOException e) {
			e.printStackTrace();
		}

		return 0.0;
	}

	public Vector readVector() {
		try {
			String[] line = m_br.readLine().split(" ");
			Vector res = new Vector(line.length);

			for (int i = 0; i < res.getSize(); ++i) {
				res.set(i, Double.parseDouble(line[i]));
			}

			return res;
		} catch (IOException e) {
			e.printStackTrace();
		}

		return null;
	}

	public Matrix readMatrix(int n) {
		Matrix res = new Matrix(n);

		for (int i = 0; i < n; ++i) {
			Vector vec = readVector();

			for (int j = 0; j < n; ++j) {
				res.set(i, j, vec.get(j));
			}
		}

		return res;
	}

	public void close() {
		try {
			m_br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private BufferedReader m_br;
}
