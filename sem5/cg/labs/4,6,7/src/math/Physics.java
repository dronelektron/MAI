package math;

import main.Camera;
import objects.Terrain;

public class Physics {
	public Physics(Camera camera, Terrain terrain) {
		this.camera = camera;
		this.terrain = terrain;

		velocity = new Vector();
		flyTime = 0.0f;
		isOnGround = false;
	}

	public void solve(float dx, float dz, float delta) {
		float deltaDist = WALK_SPEED * delta;
		float deltaX = deltaDist * dx;
		float deltaZ = deltaDist * dz;

		camera.setX(camera.getX() + deltaX);
		camera.setZ(camera.getZ() + deltaZ);

		if (!isOnGround) {
			if (camera.getY() < 0.0f) {
				camera.setY(0.0f);
				velocity.setY(0.0f);
				isOnGround = true;
			} else {
				velocity = velocity.add(new Vector(0.0f, -GRAVITY * flyTime, 0.0f, 0.0f));
				camera.setY(camera.getY() + velocity.getY() * delta);
				flyTime += delta;
			}
		}
	}

	public void makeJump() {
		if (isOnGround) {
			velocity = velocity.add(JUMP_SPEED);
			flyTime = 0.0f;
			isOnGround = false;
		}
	}

	public static final float GRAVITY = 9.81f; // units/sec^2
	public static final float WALK_SPEED = 40.0f; // units/sec
	public static final Vector JUMP_SPEED = new Vector(0.0f, 30.0f, 0.0f, 0.0f); // units/sec
	private Camera camera;
	private Terrain terrain;
	private Vector velocity;
	private float flyTime;
	private boolean isOnGround;
}
