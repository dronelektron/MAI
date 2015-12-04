package math;

import org.lwjgl.BufferUtils;
import java.nio.FloatBuffer;

public class Matrix {
	public Matrix() {
		mat = new float[4][4];
	}

	public Matrix initIdentity() {
		mat[0][0] = 1.0f; mat[0][1] = 0.0f; mat[0][2] = 0.0f; mat[0][3] = 0.0f;
		mat[1][0] = 0.0f; mat[1][1] = 1.0f; mat[1][2] = 0.0f; mat[1][3] = 0.0f;
		mat[2][0] = 0.0f; mat[2][1] = 0.0f; mat[2][2] = 1.0f; mat[2][3] = 0.0f;
		mat[3][0] = 0.0f; mat[3][1] = 0.0f; mat[3][2] = 0.0f; mat[3][3] = 1.0f;

		return this;
	}

	public Matrix initPerspective(float fov, float aspectRatio, float zNear, float zFar)
	{
		final float THF = (float)Math.tan((float)Math.toRadians(fov) / 2.0f);
		final float RANGE = zFar - zNear;
		final float X = 1.0f / THF;
		final float Y = aspectRatio / THF;
		final float A = (zFar + zNear) / RANGE;
		final float B = -2.0f * zFar * zNear / RANGE;

		mat[0][0] = X;    mat[0][1] = 0.0f; mat[0][2] = 0.0f; mat[0][3] = 0.0f;
		mat[1][0] = 0.0f; mat[1][1] = Y;    mat[1][2] = 0.0f; mat[1][3] = 0.0f;
		mat[2][0] = 0.0f; mat[2][1] = 0.0f; mat[2][2] = A;    mat[2][3] = B;
		mat[3][0] = 0.0f; mat[3][1] = 0.0f; mat[3][2] = 1.0f; mat[3][3] = 0.0f;

		return this;
	}

	public Matrix initTranslation(float x, float y, float z) {
		mat[0][0] = 1.0f; mat[0][1] = 0.0f; mat[0][2] = 0.0f; mat[0][3] = x;
		mat[1][0] = 0.0f; mat[1][1] = 1.0f; mat[1][2] = 0.0f; mat[1][3] = y;
		mat[2][0] = 0.0f; mat[2][1] = 0.0f; mat[2][2] = 1.0f; mat[2][3] = z;
		mat[3][0] = 0.0f; mat[3][1] = 0.0f; mat[3][2] = 0.0f; mat[3][3] = 1.0f;

		return this;
	}

	public Matrix initScale(float x, float y, float z) {
		mat[0][0] = x;    mat[0][1] = 0.0f; mat[0][2] = 0.0f; mat[0][3] = 0.0f;
		mat[1][0] = 0.0f; mat[1][1] = y;    mat[1][2] = 0.0f; mat[1][3] = 0.0f;
		mat[2][0] = 0.0f; mat[2][1] = 0.0f; mat[2][2] = z;    mat[2][3] = 0.0f;
		mat[3][0] = 0.0f; mat[3][1] = 0.0f; mat[3][2] = 0.0f; mat[3][3] = 1.0f;

		return this;
	}

	public Matrix initScaleUniform(float value) {
		return initScale(value, value, value);
	}

	public Matrix initRotationX(float angle) {
		final float RAD = (float)Math.toRadians(angle);
		final float COS = (float)Math.cos(RAD);
		final float SIN = (float)Math.sin(RAD);

		mat[0][0] = 1.0f; mat[0][1] = 0.0f; mat[0][2] = 0.0f; mat[0][3] = 0.0f;
		mat[1][0] = 0.0f; mat[1][1] = COS;  mat[1][2] = -SIN; mat[1][3] = 0.0f;
		mat[2][0] = 0.0f; mat[2][1] = SIN;  mat[2][2] = COS;  mat[2][3] = 0.0f;
		mat[3][0] = 0.0f; mat[3][1] = 0.0f; mat[3][2] = 0.0f; mat[3][3] = 1.0f;

		return this;
	}

	public Matrix initRotationY(float angle) {
		final float RAD = (float)Math.toRadians(angle);
		final float COS = (float)Math.cos(RAD);
		final float SIN = (float)Math.sin(RAD);

		mat[0][0] = COS;  mat[0][1] = 0.0f; mat[0][2] = SIN;  mat[0][3] = 0.0f;
		mat[1][0] = 0.0f; mat[1][1] = 1.0f; mat[1][2] = 0.0f; mat[1][3] = 0.0f;
		mat[2][0] = -SIN; mat[2][1] = 0.0f; mat[2][2] = COS;  mat[2][3] = 0.0f;
		mat[3][0] = 0.0f; mat[3][1] = 0.0f; mat[3][2] = 0.0f; mat[3][3] = 1.0f;

		return this;
	}

	public Matrix initRotationZ(float angle) {
		final float RAD = (float)Math.toRadians(angle);
		final float COS = (float)Math.cos(RAD);
		final float SIN = (float)Math.sin(RAD);

		mat[0][0] = COS;  mat[0][1] = -SIN; mat[0][2] = 0.0f; mat[0][3] = 0.0f;
		mat[1][0] = SIN;  mat[1][1] = COS;  mat[1][2] = 0.0f; mat[1][3] = 0.0f;
		mat[2][0] = 0.0f; mat[2][1] = 0.0f; mat[2][2] = 1.0f; mat[2][3] = 0.0f;
		mat[3][0] = 0.0f; mat[3][1] = 0.0f; mat[3][2] = 0.0f; mat[3][3] = 1.0f;

		return this;
	}

	public Matrix mul(Matrix m) {
		Matrix res = new Matrix();

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				res.mat[i][j] = 0.0f;

				for (int k = 0; k < 4; ++k) {
					res.mat[i][j] += mat[i][k] * m.mat[k][j];
				}
			}
		}

		return res;
	}

	public FloatBuffer toBuffer() {
		FloatBuffer buffer = BufferUtils.createFloatBuffer(16);

		buffer.put(mat[0]);
		buffer.put(mat[1]);
		buffer.put(mat[2]);
		buffer.put(mat[3]);
		buffer.flip();

		return buffer;
	}

	private float[][] mat;
}
