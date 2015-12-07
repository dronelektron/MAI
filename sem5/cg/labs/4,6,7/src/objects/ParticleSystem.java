package objects;

import org.lwjgl.opengl.GL11;
import org.newdawn.slick.opengl.Texture;
import org.newdawn.slick.opengl.TextureLoader;
import org.newdawn.slick.util.ResourceLoader;
import java.io.IOException;
import java.util.Random;
import particles.SmokeParticle;
import main.Shader;
import math.*;

public class ParticleSystem extends Entity {
	public ParticleSystem(int count) {
		posMat = new Matrix().initTranslation(0.0f, 0.0f, 0.0f);
		rnd = new Random();

		try
		{
			texture = TextureLoader.getTexture("png", ResourceLoader.getResourceAsStream("src/resources/sprites/smoke.png"));
		} catch (IOException e) {
			e.printStackTrace();
		}

		shader = new Shader("src/resources/shaders/particle");
		particles = new SmokeParticle[count];

		for (int i = 0; i < particles.length; ++i) {
			particles[i] = null;
		}
	}

	@Override
	public void compile() {
		GL11.glNewList(dispList, GL11.GL_COMPILE);
		GL11.glBegin(GL11.GL_QUADS);
		GL11.glTexCoord2f(0.0f, 0.0f);
		GL11.glVertex3f(-0.5f, 0.5f, 0.0f);
		GL11.glTexCoord2f(0.0f, 1.0f);
		GL11.glVertex3f(-0.5f, -0.5f, 0.0f);
		GL11.glTexCoord2f(1.0f, 1.0f);
		GL11.glVertex3f(0.5f, -0.5f, 0.0f);
		GL11.glTexCoord2f(1.0f, 0.0f);
		GL11.glVertex3f(0.5f, 0.5f, 0.0f);
		GL11.glEnd();
		GL11.glEndList();
	}

	@Override
	public void update(float delta) {
		for (int i = 0; i < particles.length; ++i) {
			if (particles[i] == null) {
				float size = random(3, 5) / 5.0f;
				float maxLifeTime = random(10, 25) / 10.0f;
				Vector speed = new Vector(random(-2, 2) / 10.0f, random(7, 10) / 10.0f, random(-2, 2) / 10.0f, 1.0f);

				particles[i] = new SmokeParticle(speed, size, maxLifeTime);
			} else if (particles[i].getLifeTime() >= particles[i].getMaxLifeTime()) {
				particles[i] = null;
			} else {
				particles[i].setLifeTime(particles[i].getLifeTime() + delta);
				particles[i].update(delta);
			}
		}
	}

	@Override
	public void draw(Matrix projMat, Matrix viewMat) {
		texture.bind();
		shader.bind();
		shader.setUniform1("u_sampler", 0);

		for (SmokeParticle particle : particles) {
			if (particle == null) {
				continue;
			}

			Vector pos = particle.getPosition();
			Matrix posDeltaMat = new Matrix().initTranslation(pos.getX(), pos.getY(), pos.getZ());
			Matrix scaleMat = new Matrix().initScaleUniform(particle.getSize());
			Matrix modelMat = posDeltaMat.mul(posMat);

			modelMat.set(0, 0, viewMat.get(0, 0));
			modelMat.set(0, 1, viewMat.get(1, 0));
			modelMat.set(0, 2, viewMat.get(2, 0));
			modelMat.set(1, 0, viewMat.get(0, 1));
			modelMat.set(1, 1, viewMat.get(1, 1));
			modelMat.set(1, 2, viewMat.get(2, 1));
			modelMat.set(2, 0, viewMat.get(0, 2));
			modelMat.set(2, 1, viewMat.get(1, 2));
			modelMat.set(2, 2, viewMat.get(2, 2));
			modelMat = modelMat.mul(scaleMat);

			shader.setUniformMatrix4("u_mvp", projMat.mul(viewMat).mul(modelMat).toBuffer());
			shader.setUniform1f("u_lifetime", particle.getLifeTime());
			shader.setUniform1f("u_maxlifetime", particle.getMaxLifeTime());

			GL11.glDisable(GL11.GL_DEPTH_TEST);
			GL11.glEnable(GL11.GL_BLEND);
			GL11.glBlendFunc(GL11.GL_SRC_ALPHA, GL11.GL_ONE_MINUS_SRC_ALPHA);
			GL11.glCallList(dispList);
			GL11.glEnable(GL11.GL_DEPTH_TEST);
			GL11.glDisable(GL11.GL_BLEND);
		}

		shader.unbind();
	}

	@Override
	public void delete() {
		shader.delete();
		texture.release();

		super.delete();
	}

	public void setPosition(float x, float y, float z) {
		posMat = new Matrix().initTranslation(x, y, z);
	}

	private int random(int a, int b) {
		return a + rnd.nextInt(b - a + 1);
	}

	private Matrix posMat;
	private Random rnd;
	private Texture texture;
	private Shader shader;
	private SmokeParticle[] particles;
}
