package objects;

import org.newdawn.slick.opengl.Texture;
import org.newdawn.slick.opengl.TextureLoader;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class Terrain extends Entity {
	public Terrain() {
		float step = 1.0f;
		BufferedImage bi = null;

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
				float texCoordX = (float)j / bi.getWidth();
				float texCoordY = (float)i / bi.getHeight();

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
				indices.add(offset + bi.getWidth());
				indices.add(offset + bi.getWidth() + 1);
				indices.add(offset);
				indices.add(offset + bi.getWidth() + 1);
				indices.add(offset + 1);
			}
		}

		try
		{
			terrainTexture = TextureLoader.getTexture("png", getClass().getResource("../resources/textures/terrain4.png").openStream());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void draw() {
		terrainTexture.bind();

		super.draw();
	}

	private Texture terrainTexture;
}
