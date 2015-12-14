package math;

import main.Camera;
import objects.Terrain;

public class RayTracer {
	public RayTracer(Camera camera, float rayLen) {
		this.camera = camera;
		this.rayLen = rayLen;
	}

	public Vector trace(Terrain terrain) {
		Vector orig = new Vector(camera.getX(), camera.getY(), camera.getZ(), 1.0f);
		Vector dir = Angle.toVector(camera.getPitch(), camera.getYaw() - 90.0f).mul(rayLen);
		Vector res = new Vector();

		for (int i = 0; i < terrain.getTrianglesCount(); ++i) {
			Vector v0 = terrain.getPoint(i, 0);
			Vector v1 = terrain.getPoint(i, 1);
			Vector v2 = terrain.getPoint(i, 2);

			if (isHit(orig, orig.add(dir), v0, v1, v2, res)) {
				return res;
			}
		}

		return null;
	}

	public boolean isHit(Vector r1, Vector r2, Vector v0, Vector v1, Vector v2, Vector res) {
		Vector v0v1 = v1.sub(v0);
		Vector v0v2 = v2.sub(v0);
		Vector normal = v0v1.vec(v0v2);
		float dist1 = r1.sub(v0).dot(normal);
		float dist2 = r2.sub(v0).dot(normal);

		if (dist1 * dist2 >= 0.0f) {
			return false;
		}

		if (dist1 == dist2) {
			return false;
		}

		Vector intersect = r1.add(r2.sub(r1).mul(-dist1 / (dist2 - dist1)));
		Vector edge0 = v1.sub(v0);
		Vector edge1 = v2.sub(v1);
		Vector edge2 = v0.sub(v2);
		Vector c0 = intersect.sub(v0);
		Vector c1 = intersect.sub(v1);
		Vector c2 = intersect.sub(v2);

		if (normal.dot(edge0.vec(c0)) < 0.0f) {
			return false;
		}

		if (normal.dot(edge1.vec(c1)) < 0.0f) {
			return false;
		}

		if (normal.dot(edge2.vec(c2)) < 0.0f) {
			return false;
		}

		res.copy(intersect);

		return true;
	}

	private float rayLen;
	private Camera camera;
}
