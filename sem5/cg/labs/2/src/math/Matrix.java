package math;

public class Matrix {
	public Matrix() {
		mat = new double[4][4];
	}

	public Matrix initIdentity() {
		mat[0][0] = 1.0; mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = 0.0;
		mat[1][0] = 0.0; mat[1][1] = 1.0; mat[1][2] = 0.0; mat[1][3] = 0.0;
		mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = 1.0; mat[2][3] = 0.0;
		mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;

		return this;
	}

	public Matrix initOrtho(double aspect) {
		mat[0][0] = 1.0; mat[0][1] = 0.0;    mat[0][2] = 0.0; mat[0][3] = 0.0;
		mat[1][0] = 0.0; mat[1][1] = aspect; mat[1][2] = 0.0; mat[1][3] = 0.0;
		mat[2][0] = 0.0; mat[2][1] = 0.0;    mat[2][2] = 0.0; mat[2][3] = 0.0;
		mat[3][0] = 0.0; mat[3][1] = 0.0;    mat[3][2] = 0.0; mat[3][3] = 1.0;

		return this;
	}

	public Matrix initPerspective(double fov, double aspectRatio, double zNear, double zFar)
	{
		final double THF = Math.tan(fov * Math.PI / 360.0);
		//final double RANGE = zFar - zNear;
		final double X = 1.0 / THF;
		final double Y = aspectRatio / THF;
		final double A = 1.0; //(zFar + zNear) / RANGE;
		final double B = 0.0; //-2.0 * zFar * zNear / RANGE;

		mat[0][0] = X;   mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = 0.0;
		mat[1][0] = 0.0; mat[1][1] = Y;   mat[1][2] = 0.0; mat[1][3] = 0.0;
		mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = A;   mat[2][3] = B;
		mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 1.0; mat[3][3] = 0.0;

		return this;
	}

	public Matrix initScreenSpace(double width, double height)
	{
		final double A = (width - 1.0) / 2.0;
		final double B = (height - 1.0) / 2.0;

		mat[0][0] = A;   mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = A;
		mat[1][0] = 0.0; mat[1][1] = -B;  mat[1][2] = 0.0; mat[1][3] = B;
		mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = 1.0; mat[2][3] = 0.0;
		mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;

		return this;
	}

	public Matrix initTranslation(double x, double y, double z) {
		mat[0][0] = 1.0; mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = x;
		mat[1][0] = 0.0; mat[1][1] = 1.0; mat[1][2] = 0.0; mat[1][3] = y;
		mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = 1.0; mat[2][3] = z;
		mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;

		return this;
	}

	public Matrix initScale(double x, double y, double z) {
		mat[0][0] = x;   mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = 0.0;
		mat[1][0] = 0.0; mat[1][1] = y;   mat[1][2] = 0.0; mat[1][3] = 0.0;
		mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = z;   mat[2][3] = 0.0;
		mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;

		return this;
	}

	public Matrix initRotationX(double angle) {
		final double rad = angle * Math.PI / 180.0;
		final double COS = Math.cos(rad);
		final double SIN = Math.sin(rad);

		mat[0][0] = 1.0; mat[0][1] = 0.0; mat[0][2] = 0.0;  mat[0][3] = 0.0;
		mat[1][0] = 0.0; mat[1][1] = COS; mat[1][2] = -SIN; mat[1][3] = 0.0;
		mat[2][0] = 0.0; mat[2][1] = SIN; mat[2][2] = COS;  mat[2][3] = 0.0;
		mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0;  mat[3][3] = 1.0;

		return this;
	}

	public Matrix initRotationY(double angle) {
		final double rad = angle * Math.PI / 180.0;
		final double COS = Math.cos(rad);
		final double SIN = Math.sin(rad);

		mat[0][0] = COS;  mat[0][1] = 0.0; mat[0][2] = SIN; mat[0][3] = 0.0;
		mat[1][0] = 0.0;  mat[1][1] = 1.0; mat[1][2] = 0.0; mat[1][3] = 0.0;
		mat[2][0] = -SIN; mat[2][1] = 0.0; mat[2][2] = COS; mat[2][3] = 0.0;
		mat[3][0] = 0.0;  mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;

		return this;
	}

	public Matrix initRotationZ(double angle) {
		final double rad = angle * Math.PI / 180.0;
		final double COS = Math.cos(rad);
		final double SIN = Math.sin(rad);

		mat[0][0] = COS; mat[0][1] = -SIN; mat[0][2] = 0.0; mat[0][3] = 0.0;
		mat[1][0] = SIN; mat[1][1] = COS;  mat[1][2] = 0.0; mat[1][3] = 0.0;
		mat[2][0] = 0.0; mat[2][1] = 0.0;  mat[2][2] = 1.0; mat[2][3] = 0.0;
		mat[3][0] = 0.0; mat[3][1] = 0.0;  mat[3][2] = 0.0; mat[3][3] = 1.0;

		return this;
	}

	public Matrix mul(Matrix m) {
		Matrix res = new Matrix();

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				res.mat[i][j] = 0.0;

				for (int k = 0; k < 4; ++k) {
					res.mat[i][j] += mat[i][k] * m.mat[k][j];
				}
			}
		}

		return res;
	}

	public Vector transform(Vector v) {
		double x = mat[0][0] * v.getX() + mat[0][1] * v.getY() + mat[0][2] * v.getZ() + mat[0][3] * v.getW();
		double y = mat[1][0] * v.getX() + mat[1][1] * v.getY() + mat[1][2] * v.getZ() + mat[1][3] * v.getW();
		double z = mat[2][0] * v.getX() + mat[2][1] * v.getY() + mat[2][2] * v.getZ() + mat[2][3] * v.getW();
		double w = mat[3][0] * v.getX() + mat[3][1] * v.getY() + mat[3][2] * v.getZ() + mat[3][3] * v.getW();

		return new Vector(x, y, z, w);
	}

	private double[][] mat;
}
