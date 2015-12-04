package particles;

import org.newdawn.slick.opengl.Texture;
import org.newdawn.slick.opengl.TextureLoader;
import org.newdawn.slick.util.ResourceLoader;
import java.io.IOException;
import java.util.Random;
import math.Vector;
import math.Matrix;
import main.Shader;

public class ParticleSystem {
	public ParticleSystem(int count) {
		vpMat = new Matrix().initIdentity();
		rnd = new Random();

		try
		{
			texture = TextureLoader.getTexture("png", ResourceLoader.getResourceAsStream("src/resources/sprites/smoke.png"));
		} catch (IOException e) {
			e.printStackTrace();
		}

		shader = new Shader("src/resources/shaders/particle");
		particles = new SmokeParticle[count];
		particleDL = new ParticleDL();

		for (int i = 0; i < particles.length; ++i) {
			particles[i] = null;
		}
	}

	public void delete() {
		shader.delete();
		particleDL.delete();
	}

	public void update(float delta) {
		texture.bind();

		for (int i = 0; i < particles.length; ++i) {
			shader.bind();
			shader.setUniform1("u_sampler", 0);

			if (particles[i] == null) {
				Vector speed = new Vector(random(-2, 2) / 10.0f, random(7, 10) / 10.0f, 0.0f, 1.0f);
				float size = random(3, 5) / 5.0f;
				float maxLifeTime = random(10, 25) / 10.0f;

				particles[i] = new SmokeParticle(speed, size, maxLifeTime);
			} else if (particles[i].getLifeTime() >= particles[i].getMaxLifeTime()) {
				particles[i] = null;
			} else {
				Vector particlesBasePos = new Vector(0.0f, 0.0f, 0.0f, 1.0f);
				Matrix posMat = new Matrix().initTranslation(
						particlesBasePos.getX(),
						particlesBasePos.getY(),
						particlesBasePos.getZ()
				);
				Matrix scaleMat = new Matrix().initScaleUniform(particles[i].getSize());
				Matrix modelMat = posMat.mul(scaleMat);

				shader.setUniformMatrix4("u_mvp", vpMat.mul(modelMat).toBuffer());
				shader.setUniform4("u_velocity", particles[i].getVelocity().toBuffer());
				shader.setUniform1f("u_lifetime", particles[i].getLifeTime());
				shader.setUniform1f("u_maxlifetime", particles[i].getMaxLifeTime());
				particles[i].setLifeTime(particles[i].getLifeTime() + delta);
				particleDL.draw();
			}

			shader.unbind();
		}
	}

	public void setVpMat(Matrix vpMat) {
		this.vpMat = vpMat;
	}

	private int random(int a, int b) {
		return a + rnd.nextInt(b - a + 1);
	}

	private Matrix vpMat;
	private Random rnd;
	private Texture texture;
	private Shader shader;
	private ParticleDL particleDL;
	private SmokeParticle[] particles;
}
