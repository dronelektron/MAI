package main;

import math.Matrix;

public class Camera {
	public Camera(int width, int height) {
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		pitch = 0.0f;
		yaw = 0.0f;
		moveSpeed = 40.0f;
		rotateSpeed = 80.0f;
	}

	public void move(float delta, float dir) {
		float dist = moveSpeed * delta;
		float pitchRad = (float)Math.toRadians(pitch);
		float yawRad = (float)Math.toRadians(yaw + 90.0f * dir);

		x += dist * Math.cos(yawRad) * (dir > 0.0f ? Math.cos(pitchRad) : 1.0f);
		z -= dist * Math.sin(yawRad) * (dir > 0.0f ? Math.cos(pitchRad) : 1.0f);
		y += dist * Math.sin(pitchRad * dir);
	}

	public void rotate(float deltaPitch, float deltaYaw) {
		pitch += deltaPitch * rotateSpeed;
		yaw += deltaYaw * rotateSpeed;

		if (pitch > 90.0f) {
			pitch = 90.0f;
		} else if (pitch < -90.0f) {
			pitch = -90.0f;
		}
	}

	public Matrix getView() {
		Matrix rotMatX = new Matrix().initRotationX(-pitch);
		Matrix rotMatY = new Matrix().initRotationY(-yaw);
		Matrix moveMat = new Matrix().initTranslation(-x, -y, -z);

		return rotMatX.mul(rotMatY).mul(moveMat);
	}

	public float getX() {
		return x;
	}

	public void setX(float x) {
		this.x = x;
	}

	public float getY() {
		return y;
	}

	public void setY(float y) {
		this.y = y;
	}

	public float getZ() {
		return z;
	}

	public void setZ(float z) {
		this.z = z;
	}

	public float getPitch() {
		return pitch;
	}

	public void setPitch(float pitch) {
		this.pitch = pitch;
	}

	public float getYaw() {
		return yaw;
	}

	public void setYaw(float yaw) {
		this.yaw = yaw;
	}

	public void setMoveSpeed(float speed) {
		moveSpeed = speed;
	}

	public void setRotateSpeed(float speed) {
		rotateSpeed = speed;
	}

	private float x;
	private float y;
	private float z;
	private float pitch;
	private float yaw;
	private float moveSpeed;
	private float rotateSpeed;
}
