package particles;

import math.Vector;

public class SmokeParticle {
	public SmokeParticle(Vector velocity, float size, float maxLifeTime) {
		this.velocity = velocity;
		this.size = size;
		this.maxLifeTime = maxLifeTime;
		lifeTime = 0.0f;
	}

	public Vector getVelocity() {
		return velocity;
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

	private Vector velocity;
	private float size;
	private float maxLifeTime;
	private float lifeTime;
}
