package objects;

import org.lwjgl.opengl.GL11;
import org.newdawn.slick.opengl.Texture;
import org.newdawn.slick.opengl.TextureLoader;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import main.Shader;
import math.Matrix;

public class Terrain extends Entity {
	public Terrain() {
		float step = 1.0f;
		int texSize = 64;
		BufferedImage bi = null;

		points = new ArrayList<>();
		texCoords = new ArrayList<>();
		indices = new ArrayList<>();

		try
		{
			bi = ImageIO.read(getClass().getResource("../resources/textures/terrain_hm4_v2.png"));
		} catch (IOException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < bi.getHeight(); ++i) {
			for (int j = 0; j < bi.getWidth(); ++j) {
				float x = j * step;
				float z = i * step;
				float y = (bi.getRGB(j, i) & 255) * step;
				float texCoordX = (float)j / texSize;
				float texCoordY = 1.0f - (float)i / texSize;

				points.add(x);
				points.add(y);
				points.add(z);

				texCoords.add(texCoordX);
				texCoords.add(texCoordY);
			}
		}

		for (int i = 0; i < bi.getHeight() - 1; ++i) {
			for (int j = 0; j < bi.getWidth() - 1; ++j) {
				int offset = i * bi.getWidth() + j;

				indices.add(offset);
				indices.add(offset + bi.getWidth() + 1);
				indices.add(offset + bi.getWidth());
				indices.add(offset);
				indices.add(offset + 1);
				indices.add(offset + bi.getWidth() + 1);
			}
		}

		try
		{
			texture = TextureLoader.getTexture("png", getClass().getResource("../resources/textures/terrain5.png").openStream());
		} catch (IOException e) {
			e.printStackTrace();
		}

		shader = new Shader("src/resources/shaders/terrain");
	}

	@Override
	public void compile() {
		GL11.glNewList(dispList, GL11.GL_COMPILE);
		GL11.glBegin(GL11.GL_TRIANGLES);
		GL11.glColor3f(1.0f, 1.0f, 1.0f);

		for (int i = 0; i < indices.size(); i += 3) {
			for (int j = 0; j < 3; ++j) {
				GL11.glTexCoord2f(texCoords.get(2 * indices.get(i + j)),
						texCoords.get(2 * indices.get(i + j) + 1));

				GL11.glVertex3f(
						points.get(3 * indices.get(i + j)),
						points.get(3 * indices.get(i + j) + 1),
						points.get(3 * indices.get(i + j) + 2));
			}
		}

		GL11.glEnd();
		GL11.glEndList();
	}

	@Override
	public void draw(Matrix projMat, Matrix viewMat) {
		texture.bind();
		shader.bind();
		shader.setUniform1("u_sampler", 0);
		shader.setUniformMatrix4("u_mvp", projMat.mul(viewMat).toBuffer());

		GL11.glCallList(dispList);

		shader.unbind();
	}

	@Override
	public void delete() {
		shader.delete();
		texture.release();

		super.delete();
	}

	private ArrayList<Float> points;
	private ArrayList<Float> texCoords;
	private ArrayList<Integer> indices;
	private Texture texture;
	private Shader shader;
}
