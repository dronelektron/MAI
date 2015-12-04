package main;

import org.lwjgl.LWJGLException;
import org.lwjgl.input.Keyboard;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;
import org.lwjgl.opengl.GL11;
import math.Matrix;
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

		ps.setPosition(120.0f, 130.0f, 100.0f);

		prevTime = System.nanoTime();
		delta = 0.0f;
		camera = new Camera(WIDTH, HEIGHT);
		projection = new Matrix().initPerspective(75.0f, (float)WIDTH / HEIGHT, 0.1f, 1000.0f);
		entities = new Entity[]{new Terrain(), ps};
		camera.setX(100.0f);
		camera.setY(150.0f);
		camera.setZ(50.0f);

		for (Entity ent : entities) {
			ent.compile();
		}
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
		if (Keyboard.isKeyDown(Keyboard.KEY_W)) {
			camera.move(-delta, 1.0f);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_S)) {
			camera.move(delta, 1.0f);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_A)) {
			camera.move(-delta, 0.0f);
		}

		if (Keyboard.isKeyDown(Keyboard.KEY_D)) {
			camera.move(delta, 0.0f);
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

		if (Display.wasResized()) {
			GL11.glViewport(0, 0, Display.getWidth(), Display.getHeight());

			projection = new Matrix().initPerspective(75.0f, (float)Display.getWidth() / Display.getHeight(), 0.1f, 1000.0f);
		}
	}

	private static final int FPS = 60;
	private static final String TITLE = "Компьютерная графика - лабораторная работа 4, 6, 7";

	private static long prevTime;
	private static float delta;
	private static Camera camera;
	private static Matrix projection;
	private static Entity[] entities;
}
