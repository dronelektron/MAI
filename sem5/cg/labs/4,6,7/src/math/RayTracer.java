package math;

import java.util.LinkedList;
import main.Camera;
import objects.*;

public class RayTracer {
	public RayTracer(Camera camera, Terrain terrain, LinkedList<Entity> entities) {
		this.camera = camera;
		this.terrain = terrain;
		this.entities = entities;
	}

	public void check() {
		float rayLen = 8;
		float pitch = (float)Math.toRadians(camera.getPitch());
		float yaw = (float)Math.toRadians(camera.getYaw() + 90.0f);
		Vector orig = new Vector(camera.getX(), camera.getY(), camera.getZ(), 1.0f);
		Vector dir = new Vector(
				-(float)Math.cos(yaw) * (float)Math.cos(pitch),
				-(float)Math.sin(pitch),
				(float)Math.sin(yaw) * (float)Math.cos(pitch),
				0.0f
		).mul(rayLen);

		for (int i = 0; i < terrain.getTrianglesCount(); ++i) {
			Vector v0 = terrain.getPoint(i, 0);
			Vector v1 = terrain.getPoint(i, 1);
			Vector v2 = terrain.getPoint(i, 2);

			if (isHit(orig, orig.add(dir), v0, v1, v2)) {
				System.out.println("Hit: " + i);

				ParticleSystem ps = (ParticleSystem)entities.getLast();
				
				ps.setPosition(v0.getX(), v0.getY(), v0.getZ());
			}
		}
	}

	public boolean isHit(Vector r1, Vector r2, Vector v0, Vector v1, Vector v2) {
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

		Vector res = r1.add(r2.sub(r1).mul(-dist1 / (dist2 - dist1)));
		Vector edge0 = v1.sub(v0);
		Vector edge1 = v2.sub(v1);
		Vector edge2 = v0.sub(v2);
		Vector c0 = res.sub(v0);
		Vector c1 = res.sub(v1);
		Vector c2 = res.sub(v2);

		if (normal.dot(edge0.vec(c0)) < 0.0f) {
			return false;
		}

		if (normal.dot(edge1.vec(c1)) < 0.0f) {
			return false;
		}

		if (normal.dot(edge2.vec(c2)) < 0.0f) {
			return false;
		}

		return true;
	}

	private Camera camera;
	private Terrain terrain;
	private LinkedList<Entity> entities;
}
