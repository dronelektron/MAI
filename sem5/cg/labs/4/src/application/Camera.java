package application;

import org.lwjgl.opengl.GL11;
import org.lwjgl.util.glu.GLU;

public class Camera {
	public Camera(int width, int height) {
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		pitch = 0.0f;
		yaw = 0.0f;

		moveSpeed = 80.0f;
		rotateSpeed = 90.0f;

		resizeView(width, height);
	}

	public void move(float delta, float dir) {
		float dist = moveSpeed * delta;
		float pitchRad = (float)Math.toRadians(pitch);
		float yawRad = (float)Math.toRadians(yaw + 90.0f * dir);

		x += dist * Math.cos(yawRad) * (dir > 0.0f ? Math.cos(pitchRad) : 1.0f);
		z += dist * Math.sin(yawRad) * (dir > 0.0f ? Math.cos(pitchRad) : 1.0f);
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

	public void useView() {
		GL11.glRotatef(pitch, 1.0f, 0.0f, 0.0f);
		GL11.glRotatef(yaw, 0.0f, 1.0f, 0.0f);
		GL11.glTranslatef(x, y, z);
	}

	public void setPos(float x, float y, float z) {
		this.x = -x;
		this.y = -y;
		this.z = -z;
	}

	public void resizeView(int width, int height) {
		GL11.glMatrixMode(GL11.GL_PROJECTION);
		GL11.glLoadIdentity();
		GLU.gluPerspective(70.0f, (float)width / height, 0.3f, 500.0f);
		GL11.glMatrixMode(GL11.GL_MODELVIEW);
		GL11.glViewport(0, 0, width, height);
	}

	public void setX(float x) {
		this.x = x;
	}

	public void setY(float y) {
		this.y = y;
	}

	public void setZ(float z) {
		this.z = z;
	}

	public void setPitch(float pitch) {
		this.pitch = pitch;
	}

	public void setRY(float yaw) {
		this.yaw = yaw;
	}

	public void setMoveSpeed(float speed) {
		moveSpeed = speed;
	}

	public void setRotateSpeed(float speed) {
		rotateSpeed = speed;
	}

	public float getX() {
		return x;
	}

	public float getY() {
		return y;
	}

	public float getZ() {
		return z;
	}

	public float getPitch() {
		return pitch;
	}

	public float getYaw() {
		return yaw;
	}

	private float x;
	private float y;
	private float z;
	private float pitch;
	private float yaw;

	private float moveSpeed;
	private float rotateSpeed;
}
