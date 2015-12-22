package main;

import org.lwjgl.LWJGLException;
import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.*;
import math.Matrix;
import math.Vector;
import math.Physics;
import math.RayTracer;
import objects.*;

public class Main {
	public static void main(String[] args) {
		initDisplay();
		initGL();
		loop();
		delete();
	}

	private static void initDisplay() {
		final int WIDTH = 1024;
		final int HEIGHT = 768;

		try {
			Display.setDisplayMode(new DisplayMode(WIDTH, HEIGHT));
			Display.setTitle(TITLE);
			//Display.setResizable(true);
			Display.create();
		} catch (LWJGLException e) {
			e.printStackTrace();
		}

		ParticleSystem ps = new ParticleSystem(50);
		Terrain terrain = new Terrain();

		ps.setPosition(128.0f, 8.0f, -128.0f);

		isFlyMode = false;
		isMouseClicked = false;
		prevTime = System.nanoTime();
		delta = 0.0f;
		entities = new Entity[] {terrain, new Cylinder(), ps};
		fbos = new WaterFBO();
		water = new Water(fbos, 4.0f);
		camera = new Camera();
		projection = new Matrix().initPerspective(75.0f, (float)WIDTH / HEIGHT, 0.1f, 500.0f);
		physics = new Physics(camera, terrain);
		rayTracer = new RayTracer(camera, 16.0f);

		camera.setX(128.0f);
		camera.setZ(-128.0f);
		/*
		camera.setX(150.0f);
		camera.setY(8.0f);
		camera.setZ(-134.0f);
		camera.setPitch(25.0f);
		camera.setYaw(-50.0f);
		*/
		for (Entity ent : entities) {
			ent.compile();
		}

		water.compile();
	}

	private static void initGL() {
		GL11.glEnable(GL11.GL_DEPTH_TEST);
		GL11.glEnable(GL11.GL_CULL_FACE);
		GL11.glEnable(GL11.GL_TEXTURE_2D);
	}

	private static void loop() {
		while (!Display.isCloseRequested()) {
			updateDelta();
			getInput();

			if (!isFlyMode) {
				physics.solve(speedX, speedZ, delta);
			}

			float distance = 2.0f * (camera.getY() - water.getHeight());
			Vector cameraPos = new Vector(camera.getX(), camera.getY(), camera.getZ(), 1.0f);

			GL11.glEnable(GL30.GL_CLIP_DISTANCE0);

			fbos.bindReflectionFrameBuffer();
			camera.setY(camera.getY() - distance);
			camera.setPitch(-camera.getPitch());
			draw(new Vector(0.0f, 1.0f, 0.0f, -water.getHeight()), cameraPos);
			camera.setY(camera.getY() + distance);
			camera.setPitch(-camera.getPitch());

			fbos.bindRefractionFrameBuffer();
			draw(new Vector(0.0f, -1.0f, 0.0f, water.getHeight()), cameraPos);

			fbos.unbindCurrentFrameBuffer();

			GL11.glDisable(GL30.GL_CLIP_DISTANCE0);

			draw(new Vector(0.0f, -1.0f, 0.0f, 1000.0f), cameraPos);
			water.update(delta);
			water.draw(projection, camera.getView(), null, cameraPos);

			Display.update();
			Display.sync(FPS);
		}
	}

	private static void delete() {
		for (Entity ent : entities) {
			ent.delete();
		}

		water.delete();
		fbos.delete();

		Display.destroy();
	}

	private static void draw(Vector plane, Vector cameraPos) {
		Matrix view = camera.getView();

		GL11.glClearColor(0.4f, 0.7f, 0.8f, 1.0f);
		GL11.glClear(GL11.GL_COLOR_BUFFER_BIT | GL11.GL_DEPTH_BUFFER_BIT);
		GL13.glActiveTexture(GL13.GL_TEXTURE0);

		for (Entity ent : entities) {
			ent.update(delta);
			ent.draw(projection, view, plane, cameraPos);
		}
	}

	private static void updateDelta() {
		long curTime = System.nanoTime();

		delta = (curTime - prevTime) / 1000000000.0f;
		prevTime = curTime;
	}

	private static void getInput() {
		float yaw = (float)Math.toRadians(camera.getYaw());

		speedX = 0.0f;
		speedZ = 0.0f;

		if (Keyboard.isKeyDown(Keyboard.KEY_W)) {
			if (isFlyMode) {
				camera.move(delta, true);
			} else {
				float yawOffset = (float)Math.toRadians(90.0f);
				float dx = (float)Math.cos(yaw + yawOffset);
				float dz = (float)Math.sin(yaw + yawOffset);

				speedX += -dx;
				speedZ += dz;
			}
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_S)) {
			if (isFlyMode) {
				camera.move(-delta, true);
			} else {
				float yawOffset = (float)Math.toRadians(90.0f);
				float dx = (float)Math.cos(yaw + yawOffset);
				float dz = (float)Math.sin(yaw + yawOffset);

				speedX += dx;
				speedZ += -dz;
			}
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_A)) {
			if (isFlyMode) {
				camera.move(-delta, false);
			} else {
				float dx = (float)Math.cos(yaw);
				float dz = (float)Math.sin(yaw);

				speedX += -dx;
				speedZ += dz;
			}
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_D)) {
			if (isFlyMode) {
				camera.move(delta, false);
			} else {
				float dx = (float)Math.cos(yaw);
				float dz = (float)Math.sin(yaw);

				speedX += dx;
				speedZ += -dz;
			}
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_UP)) {
			camera.rotate(-delta, 0.0f);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_DOWN)) {
			camera.rotate(delta, 0.0f);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_LEFT)) {
			camera.rotate(0.0f, -delta);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_RIGHT)) {
			camera.rotate(0.0f, delta);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_SPACE)) {
			physics.makeJump();
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_Z)) {
			Mouse.setGrabbed(true);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_X)) {
			Mouse.setGrabbed(false);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_F)) {
			isFlyMode = true;
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_G)) {
			isFlyMode = false;
		}

		if (Mouse.isButtonDown(0) && !isMouseClicked) {
			isMouseClicked = true;

			Vector res = rayTracer.trace((Terrain)entities[0]);

			if (res != null) {
				ParticleSystem ps = (ParticleSystem)entities[2];

				ps.setPosition(res.getX(), res.getY(), res.getZ());
			}
		}

		if (!Mouse.isButtonDown(0) && isMouseClicked) {
			isMouseClicked = false;
		}

		if (Display.wasResized()) {
			GL11.glViewport(0, 0, Display.getWidth(), Display.getHeight());

			projection = new Matrix().initPerspective(75.0f, (float)Display.getWidth() / Display.getHeight(), 0.1f, 500.0f);
		}

		if (Mouse.isGrabbed()) {
			int dx = Mouse.getDX();
			int dy = Mouse.getDY();
			float angle = delta * MOUSE_SENS;

			camera.rotate(-dy * angle, dx * angle);
		}
	}

	private static final float MOUSE_SENS = 0.6f;
	private static final int FPS = 300;
	private static final String TITLE = "Компьютерная графика - курсовая работа";

	private static boolean isFlyMode;
	private static boolean isMouseClicked;
	private static long prevTime;
	private static float delta;
	private static float speedX;
	private static float speedZ;
	private static Camera camera;
	private static Matrix projection;
	private static Physics physics;
	private static RayTracer rayTracer;
	private static Water water;
	private static WaterFBO fbos;
	private static Entity[] entities;
}
