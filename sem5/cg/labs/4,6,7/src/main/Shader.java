package main;

import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL20;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;

public class Shader {
	public Shader(String fileName) {
		final String VS_EXT = ".vert";
		final String FS_EXT = ".frag";
		StringBuilder vertexShaderSrc = new StringBuilder();
		StringBuilder fragmentShaderSrc = new StringBuilder();

		shaderProgram = GL20.glCreateProgram();
		vertexShader = GL20.glCreateShader(GL20.GL_VERTEX_SHADER);
		fragmentShader = GL20.glCreateShader(GL20.GL_FRAGMENT_SHADER);

		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName + VS_EXT));
			String line;

			while ((line = reader.readLine()) != null) {
				vertexShaderSrc.append(line).append('\n');
			}

			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName + FS_EXT));
			String line;

			while ((line = reader.readLine()) != null) {
				fragmentShaderSrc.append(line).append('\n');
			}

			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		GL20.glShaderSource(vertexShader, vertexShaderSrc);
		GL20.glShaderSource(fragmentShader, fragmentShaderSrc);
		GL20.glCompileShader(vertexShader);
		GL20.glCompileShader(fragmentShader);
		GL20.glAttachShader(shaderProgram, vertexShader);
		GL20.glAttachShader(shaderProgram, fragmentShader);
		GL20.glLinkProgram(shaderProgram);
		GL20.glValidateProgram(shaderProgram);

		if (GL20.glGetShaderi(vertexShader, GL20.GL_COMPILE_STATUS) == GL11.GL_FALSE) {
			System.out.println("Vertex shader compilation failed");
		}

		if (GL20.glGetShaderi(fragmentShader, GL20.GL_COMPILE_STATUS) == GL11.GL_FALSE) {
			System.out.println("Fragment shader compilation failed");
		}
	}

	public void setUniformMatrix4(String name, FloatBuffer mat) {
		int index = GL20.glGetUniformLocation(shaderProgram, name);

		GL20.glUniformMatrix4(index, true, mat);
	}

	public void setUniform4(String name, FloatBuffer vec) {
		int index = GL20.glGetUniformLocation(shaderProgram, name);

		GL20.glUniform4(index, vec);
	}

	public void setUniform1(String name, int value) {
		int index = GL20.glGetUniformLocation(shaderProgram, name);

		GL20.glUniform1i(index, value);
	}

	public void setUniform1f(String name, float value) {
		int index = GL20.glGetUniformLocation(shaderProgram, name);

		GL20.glUniform1f(index, value);
	}

	public void bind() {
		GL20.glUseProgram(shaderProgram);
	}

	public void unbind() {
		GL20.glUseProgram(0);
	}

	public void delete() {
		GL20.glDetachShader(shaderProgram, fragmentShader);
		GL20.glDetachShader(shaderProgram, vertexShader);
		GL20.glDeleteProgram(shaderProgram);
		GL20.glDeleteShader(fragmentShader);
		GL20.glDeleteShader(vertexShader);
	}

	private int shaderProgram;
	private int vertexShader;
	private int fragmentShader;
}
