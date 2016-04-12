package libnm.util;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Logger {
	public Logger(String fileName) {
		try {
			m_br = new BufferedWriter(new FileWriter(fileName));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void close() {
		try {
			m_br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void write(String text) {
		try {
			m_br.write(text);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void writeln() {
		try {
			m_br.write("\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void writeln(String text) {
		try {
			m_br.write(text + "\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private BufferedWriter m_br;
}
