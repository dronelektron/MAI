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

		float y = terrain.getY(camera.getX(), camera.getZ()) + PLAYER_TALL;

		if (!isOnGround) {
			if (camera.getY() < y) {
				camera.setY(y);
				velocity.setY(0.0f);
				isOnGround = true;
			} else {
				camera.setY(camera.getY() + velocity.getY() * delta);
				velocity.setY(JUMP_SPEED - GRAVITY * flyTime);
				flyTime += delta;
			}
		} else {
			camera.setY(y);
		}
	}

	public void makeJump() {
		if (isOnGround) {
			velocity.setY(JUMP_SPEED);
			flyTime = 0.0f;
			isOnGround = false;
		}
	}

	public static final float GRAVITY = 9.81f; // units/sec^2
	public static final float WALK_SPEED = 15.0f; // units/sec
	public static final float JUMP_SPEED = 5.0f; // units/sec
	public static final float PLAYER_TALL = 1.0f; // units
	private Camera camera;
	private Terrain terrain;
	private Vector velocity;
	private float flyTime;
	private boolean isOnGround;
}
