package particles;

import math.Vector;

public class SmokeParticle {
	public SmokeParticle(Vector velocity, float size, float maxLifeTime) {
		this.position = new Vector(0.0f, 0.0f, 0.0f, 1.0f);
		this.velocity = velocity;
		this.size = size;
		this.maxLifeTime = maxLifeTime;
		lifeTime = 0.0f;
	}

	public void update(float delta) {
		position = position.add(velocity.mul(delta));
	}

	public Vector getPosition() {
		return position;
	}

	public float getLifeTime() {
		return lifeTime;
	}

	public void setLifeTime(float lifeTime) {
		this.lifeTime = lifeTime;
	}

	public float getSize() {
		return size;
	}

	public float getMaxLifeTime() {
		return maxLifeTime;
	}

	private Vector position;
	private Vector velocity;
	private float size;
	private float maxLifeTime;
	private float lifeTime;
}
