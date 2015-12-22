package objects;

import java.util.ArrayList;
import org.lwjgl.opengl.GL11;
import org.newdawn.slick.opengl.Texture;
import main.TextureLoader;
import main.Shader;
import math.Matrix;
import math.Vector;

public class Cylinder extends Entity {
	public Cylinder() {
		int sides = 6;
		int mod = sides * 2;
		float step = (float)Math.PI * 2.0f / sides;
		String textureImg = "src/resources/textures/cylinder.png";

		points = new ArrayList<>();
		texCoords = new ArrayList<>();
		indices = new ArrayList<>();

		for (int i = 0; i < sides; ++i) {
			float angle = i * step;
			float x = (float)Math.cos(angle);
			float z = (float)Math.sin(angle);

			points.add(new Vector(x, 0.0f, z, 1.0f));
			points.add(new Vector(x, 1.0f, z, 1.0f));
		}

		points.add(new Vector(0.0f, 0.0f, 0.0f, 1.0f));
		points.add(new Vector(0.0f, 1.0f, 0.0f, 1.0f));

		for (int i = 0; i < sides; ++i) {
			int offsetPoint = i * 2;

			indices.add(offsetPoint + 1);
			indices.add(offsetPoint);
			indices.add((offsetPoint + 2) % mod);
			indices.add((offsetPoint + 2) % mod);
			indices.add((offsetPoint + 3) % mod);
			indices.add(offsetPoint + 1);

			texCoords.add(new Vector(0.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(0.0f, 1.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 1.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 1.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(0.0f, 0.0f, 0.0f, 0.0f));
		}

		for (int i = 0; i < sides; ++i) {
			indices.add(points.size() - 2);
			indices.add((i * 2 + 2) % mod);
			indices.add(i * 2);
		}

		for (int i = 0; i < sides; i += 2) {
			texCoords.add(new Vector(0.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 1.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(0.0f, 1.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(0.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 1.0f, 0.0f, 0.0f));
		}

		for (int i = 0; i < sides; ++i) {
			indices.add(points.size() - 1);
			indices.add(i * 2 + 1);
			indices.add((i * 2 + 3) % mod);
		}

		for (int i = 0; i < sides; i += 2) {
			texCoords.add(new Vector(0.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 1.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(0.0f, 1.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(0.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 0.0f, 0.0f, 0.0f));
			texCoords.add(new Vector(1.0f, 1.0f, 0.0f, 0.0f));
		}

		texture = TextureLoader.getTexture(textureImg);
		shader = new Shader("src/resources/shaders/cylinder");
		modelMat = new Matrix().initTranslation(132.0f, 5.0f, -128.0f);
	}

	@Override
	public void compile() {
		GL11.glNewList(dispList, GL11.GL_COMPILE);
		GL11.glBegin(GL11.GL_TRIANGLES);

		for (int i = 0; i < indices.size(); i += 3) {
			for (int j = 0; j < 3; ++j) {
				int offset1 = indices.get(i + j);
				int offset2 = i + j;
				Vector point = points.get(offset1);
				Vector tex = texCoords.get(offset2);

				GL11.glTexCoord2f(tex.getX(), tex.getY());
				GL11.glVertex3f(point.getX(), point.getY(), point.getZ());
			}
		}

		GL11.glEnd();
		GL11.glEndList();
	}

	@Override
	public void draw(Matrix projMat, Matrix viewMat, Vector plane, Vector cameraPos) {
		texture.bind();
		shader.bind();
		shader.setUniform1("u_sampler", 0);
		shader.setUniformMatrix4("u_viewProj", projMat.mul(viewMat).toBuffer());
		shader.setUniformMatrix4("u_model", modelMat.toBuffer());
		shader.setUniform4("u_plane", plane.toBuffer());

		GL11.glCallList(dispList);

		shader.unbind();
	}

	@Override
	public void delete() {
		shader.delete();
		texture.release();

		super.delete();
	}

	private ArrayList<Vector> points;
	private ArrayList<Vector> texCoords;
	private ArrayList<Integer> indices;
	private Texture texture;
	private Shader shader;
	private Matrix modelMat;
}
