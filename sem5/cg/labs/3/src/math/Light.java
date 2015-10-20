package math;

import javafx.scene.paint.Color;

public class Light {
	public Light(Color ambient, Color diffuse, Color specular, double specPow) {
		this.ambient = ambient;
		this.diffuse = diffuse;
		this.specular = specular;
		this.specPow = specPow;
	}

	public Color getColor(Vector normal, Vector light, Vector eye) {
		normal = normal.normalize();
		light = light.normalize();
		eye = eye.normalize();

		Vector half = light.add(eye).normalize();
		double red = 0.0;
		double green = 0.0;
		double blue = 0.0;
		double diff = Math.max(0.0, normal.dot(light));
		double spec = Math.pow(Math.max(0.0, normal.dot(half)), specPow);

		red += ambient.getRed();
		red += diffuse.getRed() * diff;
		red += specular.getRed() * spec;
		red = (red > 1) ? 1 : red;
		red = (red < 0) ? 0 : red;

		green += ambient.getGreen();
		green += diffuse.getGreen() * diff;
		green += specular.getGreen() * spec;
		green = (green > 1) ? 1 : green;
		green = (green < 0) ? 0 : green;

		blue += ambient.getBlue();
		blue += diffuse.getBlue() * diff;
		blue += specular.getBlue() * spec;
		blue = (blue > 1) ? 1 : blue;
		blue = (blue < 0) ? 0 : blue;

		return new Color(red, green, blue, 1.0);
	}

	public void setAmbient(Color ambient) {
		this.ambient = ambient;
	}

	public void setDiffuse(Color diffuse) {
		this.diffuse = diffuse;
	}

	public void setSpecular(Color specular) {
		this.specular = specular;
	}

	public void setSpecPow(double specPow) {
		this.specPow = specPow;
	}

	private Color ambient;
	private Color diffuse;
	private Color specular;
	private double specPow;
}
