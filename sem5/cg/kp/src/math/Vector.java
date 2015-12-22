package math;

import org.lwjgl.BufferUtils;
import java.nio.FloatBuffer;

public class Vector {
	public Vector() {
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		w = 0.0f;
	}

	public Vector(float x, float y, float z, float w) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}

	public Vector copy(Vector v) {
		x = v.x;
		y = v.y;
		z = v.z;
		w = v.w;

		return this;
	}

	public void setX(float x) {
		this.x = x;
	}

	public float getX() {
		return x;
	}

	public void setY(float y) {
		this.y = y;
	}

	public float getY() {
		return y;
	}

	public void setZ(float z) {
		this.z = z;
	}

	public float getZ() {
		return z;
	}

	public void setW(float w) {
		this.w = w;
	}

	public float getW() {
		return w;
	}

	public float dot(Vector v) {
		return x * v.x + y * v.y + z * v.z;
	}

	public float length() {
		return (float)Math.sqrt(x * x + y * y + z * z);
	}

	public Vector add(Vector v) {
		return new Vector(x + v.x, y + v.y, z + v.z, w + v.w);
	}

	public Vector sub(Vector v) {
		return new Vector(x - v.x, y - v.y, z - v.z, w - v.w);
	}

	public Vector mul(float a) {
		return new Vector(x * a, y * a, z * a, w * a);
	}

	public Vector vec(Vector v) {
		return new Vector(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x, w);
	}

	public Vector normalize() {
		float len = length();

		return new Vector(x / len, y / len, z / len, w);
	}

	public FloatBuffer toBuffer() {
		FloatBuffer buffer = BufferUtils.createFloatBuffer(4);

		buffer.put(x);
		buffer.put(y);
		buffer.put(z);
		buffer.put(w);
		buffer.flip();

		return buffer;
	}

	private float x;
	private float y;
	private float z;
	private float w;
}
