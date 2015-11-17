package objects;

import org.newdawn.slick.opengl.Texture;
import org.newdawn.slick.opengl.TextureLoader;
import java.io.IOException;

public class Tree extends Entity {
	public Tree() {
		// Генерация основания
		int sides = 36;
		float radius = 5.0f;
		float heightBottom = 10.0f;
		float step = 2.0f * (float)Math.PI / sides;
		float radCoef = 2.0f;
		float heightTop = 40.0f;

		// Для простого зацикливания текстуры введена еще одна точка и пара текстурных координат

		for (int i = 0; i <= sides; ++i) {
			float curAngle = i * step;
			float x = (float)Math.cos(curAngle) * radius;
			float z = (float)Math.sin(curAngle) * radius;

			points.add(x);
			points.add(0.0f);
			points.add(z);
			points.add(x);
			points.add(heightBottom);
			points.add(z);

			float texCoordX = (float)i / sides;

			texCoords.add(texCoordX);
			texCoords.add(0.0f);
			texCoords.add(texCoordX);
			texCoords.add(1.0f);
		}

		for (int i = 0; i < sides; ++i) {
			int offsetPoint = i * 2;

			indices.add(offsetPoint + 1);
			indices.add(offsetPoint + 3);
			indices.add(offsetPoint + 2);
			indices.add(offsetPoint + 1);
			indices.add(offsetPoint + 2);
			indices.add(offsetPoint);
		}

		try
		{
			topTexture = TextureLoader.getTexture("png", getClass().getResource("../resources/textures/green_top1.png").openStream());
			bottomTexture = TextureLoader.getTexture("png", getClass().getResource("../resources/textures/wood_bottom1.png").openStream());
		} catch (IOException e) {
			e.printStackTrace();
		}

		// TODO: Генерация верхушки
		/*
		for (int i = 0; i <= sides; ++i) {
			float curAngle = i * step;
			float x = (float)Math.cos(curAngle) * radius * radCoef;
			float z = (float)Math.sin(curAngle) * radius * radCoef;

			points.add(x);
			points.add(heightBottom);
			points.add(z);

			texCoords.add((float)i / sides);
			texCoords.add(1.0f);
		}

		points.add(0.0f);
		points.add(heightTop);
		points.add(0.0f);

		texCoords.add(0.0f);
		texCoords.add(0.0f);

		for (int i = 0; i < sides; ++i) {
			indices.add(points.size() - 1);
			indices.add(i + 1);
			indices.add(i);
		}
		*/
		//System.out.println((points.size() / 3) + " " + (texCoords.size() / 2));
	}

	public void draw() {
		bottomTexture.bind();

		super.draw();
	}

	private Texture topTexture;
	private Texture bottomTexture;
}
