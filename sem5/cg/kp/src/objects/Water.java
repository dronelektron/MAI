package objects;

import main.TextureLoader;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL13;
import main.Shader;
import main.WaterFBO;
import math.Matrix;
import math.Vector;
import org.newdawn.slick.opengl.Texture;

public class Water extends Entity {
	public Water(WaterFBO fbos, float height) {
		this.fbos = fbos;
		this.height = height;
		sun = new Vector(0.0f, 64.0f, 0.0f, 1.0f);
		moveFactor = 0.0f;
		shader = new Shader("src/resources/shaders/water");
		dudvTexture = TextureLoader.getTexture("src/resources/textures/water_dudv_map.png");
		normalTexture = TextureLoader.getTexture("src/resources/textures/water_normal_map.png");
		modelMat = new Matrix().initTranslation(128.0f, height, -128.0f);
		modelMat = modelMat.mul(new Matrix().initScaleUniform(128.0f));
	}

	@Override
	public void compile() {
		GL11.glNewList(dispList, GL11.GL_COMPILE);
		GL11.glBegin(GL11.GL_QUADS);

		GL11.glTexCoord2f(0.0f, 0.0f);
		GL11.glVertex3f(-1.0f, 0.0f, 1.0f);
		GL11.glTexCoord2f(0.0f, 1.0f);
		GL11.glVertex3f(-1.0f, 0.0f, -1.0f);
		GL11.glTexCoord2f(1.0f, 1.0f);
		GL11.glVertex3f(1.0f, 0.0f, -1.0f);
		GL11.glTexCoord2f(1.0f, 0.0f);
		GL11.glVertex3f(1.0f, 0.0f, 1.0f);

		GL11.glEnd();
		GL11.glEndList();
	}

	@Override
	public void draw(Matrix projMat, Matrix viewMat, Vector plane, Vector cameraPos) {
		GL13.glActiveTexture(GL13.GL_TEXTURE0);
		GL11.glBindTexture(GL11.GL_TEXTURE_2D, fbos.getReflectionTexture());
		GL13.glActiveTexture(GL13.GL_TEXTURE1);
		GL11.glBindTexture(GL11.GL_TEXTURE_2D, fbos.getRefractionTexture());
		GL13.glActiveTexture(GL13.GL_TEXTURE2);
		GL11.glBindTexture(GL11.GL_TEXTURE_2D, dudvTexture.getTextureID());
		GL13.glActiveTexture(GL13.GL_TEXTURE3);
		GL11.glBindTexture(GL11.GL_TEXTURE_2D, normalTexture.getTextureID());

		shader.bind();
		shader.setUniformMatrix4("u_viewProj", projMat.mul(viewMat).toBuffer());
		shader.setUniformMatrix4("u_model", modelMat.toBuffer());
		shader.setUniform1("u_reflectionTexture", 0);
		shader.setUniform1("u_refractionTexture", 1);
		shader.setUniform1("u_dudvMap", 2);
		shader.setUniform1("u_normalMap", 3);
		shader.setUniform4("u_cameraPos", cameraPos.toBuffer());
		shader.setUniform4("u_lightPos", new Vector(sun.getX(), sun.getY(), sun.getZ(), 1.0f).toBuffer());
		shader.setUniform1f("u_moveFactor", moveFactor);

		GL11.glCallList(dispList);

		shader.unbind();
	}

	@Override
	public void update(float delta) {
		moveFactor += WAVE_SPEED * delta;
		moveFactor %= 1.0f;
	}

	@Override
	public void delete() {
		shader.delete();
		dudvTexture.release();
		normalTexture.release();

		super.delete();
	}

	public float getHeight() {
		return height;
	}

	private static float WAVE_SPEED = 0.03f;

	private float moveFactor;
	private float height;
	private Vector sun;
	private WaterFBO fbos;
	private Shader shader;
	private Texture dudvTexture;
	private Texture normalTexture;
	private Matrix modelMat;
}
