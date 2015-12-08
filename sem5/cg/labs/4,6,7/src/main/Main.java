package main;

import math.RayTracer;
import org.lwjgl.LWJGLException;
import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.*;
import java.util.LinkedList;
import math.Matrix;
import math.Physics;
import objects.*;

public class Main {
	public static void main(String[] args) {
		initDisplay();
		initGL();
		loop();
		delete();
	}

	private static void initDisplay() {
		final int WIDTH = 800;
		final int HEIGHT = 600;

		try {
			Display.setDisplayMode(new DisplayMode(WIDTH, HEIGHT));
			Display.setTitle(TITLE);
			Display.setResizable(true);
			Display.create();
		} catch (LWJGLException e) {
			e.printStackTrace();
		}

		ParticleSystem ps = new ParticleSystem(50);
		Terrain terrain = new Terrain();

		ps.setPosition(128.0f, 16.0f, -128.0f);

		isMouseClicked = false;
		prevTime = System.nanoTime();
		delta = 0.0f;
		camera = new Camera(WIDTH, HEIGHT);
		projection = new Matrix().initPerspective(75.0f, (float)WIDTH / HEIGHT, 0.1f, 1000.0f);
		entities = new LinkedList<>();
		entities.add(terrain);
		entities.add(ps);
		physics = new Physics(camera, terrain);
		rayTracer = new RayTracer(camera, terrain, entities);
		camera.setX(128.0f);
		camera.setZ(-128.0f);

		for (Entity ent : entities) {
			ent.compile();
		}
	}

	private static void initGL() {
		GL11.glEnable(GL11.GL_DEPTH_TEST);
		GL11.glEnable(GL11.GL_CULL_FACE);
		GL11.glEnable(GL11.GL_TEXTURE_2D);
		GL30.glGenerateMipmap(GL11.GL_TEXTURE_2D);
		GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR_MIPMAP_LINEAR);
		GL11.glTexParameterf(GL11.GL_TEXTURE_2D, GL14.GL_TEXTURE_LOD_BIAS, -1.0f);
	}

	private static void loop() {
		while (!Display.isCloseRequested()) {
			updateDelta();
			getInput();
			//physics.solve(speedX, speedZ, delta);
			draw();

			Display.update();
			Display.sync(FPS);
		}
	}

	private static void delete() {
		for (Entity ent : entities) {
			ent.delete();
		}

		Display.destroy();
	}

	private static void draw() {
		Matrix view = camera.getView();

		GL11.glClearColor(0.0f, 0.5f, 1.0f, 1.0f);
		GL11.glClear(GL11.GL_COLOR_BUFFER_BIT | GL11.GL_DEPTH_BUFFER_BIT);

		for (Entity ent : entities) {
			ent.update(delta);
			ent.draw(projection, view);
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
			camera.move(-delta, 1.0f);
			float yawOffset = (float)Math.toRadians(90.0f);
			float dx = (float)Math.cos(yaw + yawOffset);
			float dz = (float)Math.sin(yaw + yawOffset);

			speedX += -dx;
			speedZ += dz;
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_S)) {
			camera.move(delta, 1.0f);
			float yawOffset = (float)Math.toRadians(90.0f);
			float dx = (float)Math.cos(yaw + yawOffset);
			float dz = (float)Math.sin(yaw + yawOffset);

			speedX += dx;
			speedZ += -dz;
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_A)) {
			camera.move(-delta, 0.0f);
			float dx = (float)Math.cos(yaw);
			float dz = (float)Math.sin(yaw);

			speedX += -dx;
			speedZ += dz;
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_D)) {
			camera.move(delta, 0.0f);
			float dx = (float)Math.cos(yaw);
			float dz = (float)Math.sin(yaw);

			speedX += dx;
			speedZ += -dz;
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

		if (Mouse.isButtonDown(0) && !isMouseClicked) {
			isMouseClicked = true;

			rayTracer.check();
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

			if (dx < -1) {
				dx = -1;
			} else if (dx > 1) {
				dx = 1;
			}

			if (dy < -1) {
				dy = -1;
			} else if (dy > 1) {
				dy = 1;
			}

			camera.rotate(-dy * delta, dx * delta);
		}
	}

	private static final int FPS = 300;
	private static final String TITLE = "Компьютерная графика - лабораторная работа 4, 6, 7";

	private static boolean isMouseClicked;
	private static long prevTime;
	private static float delta;
	private static float speedX;
	private static float speedZ;
	private static Camera camera;
	private static Matrix projection;
	private static LinkedList<Entity> entities;
	private static Physics physics;
	private static RayTracer rayTracer;
}
