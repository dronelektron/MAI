package main;

import org.lwjgl.LWJGLException;
import org.lwjgl.input.Keyboard;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;
import org.lwjgl.opengl.GL11;
import objects.*;
import math.Matrix;
import particles.ParticleSystem;

public class Main {
	public static void main(String[] args) {
		initDisplay();
		initGL();
		loop();
		delete();
	}

	public static void initDisplay() {
		try {
			Display.setDisplayMode(new DisplayMode(WIDTH, HEIGHT));
			Display.setTitle(TITLE);
			Display.setResizable(true);
			Display.create();
		} catch (LWJGLException e) {
			e.printStackTrace();
		}
	}

	public static void initGL() {
		GL11.glEnable(GL11.GL_DEPTH_TEST);
		GL11.glEnable(GL11.GL_CULL_FACE);
		GL11.glEnable(GL11.GL_TEXTURE_2D);
		GL11.glEnable(GL11.GL_BLEND);
		GL11.glBlendFunc(GL11.GL_SRC_ALPHA, GL11.GL_ONE_MINUS_SRC_ALPHA);
	}

	public static void loop() {
		/*
		Matrix view = new Matrix().initTranslation(0.0f, 0.0f, 5.0f);
		Matrix projection = new Matrix().initPerspective(
				75.0f,
				(float)WIDTH / HEIGHT,
				0.1f,
				20.0f
		);
		Matrix vp = projection.mul(view);
		ParticleSystem ps = new ParticleSystem(50);
		Camera camera = new Camera(WIDTH, HEIGHT);
		Entity[] ents = new Entity[] {new Terrain(), new Tree()};

		for (int i = 0; i < ents.length; ++i) {
			ents[i].compile();
		}

		camera.setPos(256.0f, 256.0f, 256.0f);
		ps.setVpMat(vp);

		long prevTime = System.nanoTime();
		float time = 0.0f;

		while (!Display.isCloseRequested()) {
			long curTime = System.nanoTime();
			float delta = (curTime - prevTime) / 1000000000.0f;

			prevTime = curTime;
			time += delta;

			if (Keyboard.isKeyDown(Keyboard.KEY_W)) {
				camera.move(delta, 1.0f);
			}

			if (Keyboard.isKeyDown(Keyboard.KEY_S)) {
				camera.move(-delta, 1.0f);
			}

			if (Keyboard.isKeyDown(Keyboard.KEY_A)) {
				camera.move(delta, 0.0f);
			}

			if (Keyboard.isKeyDown(Keyboard.KEY_D)) {
				camera.move(-delta, 0.0f);
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
				camera.resizeView(Display.getWidth(), Display.getHeight());
			}

			GL11.glClearColor(0.0f, 0.5f, 1.0f, 1.0f);
			GL11.glClear(GL11.GL_COLOR_BUFFER_BIT | GL11.GL_DEPTH_BUFFER_BIT);
			GL11.glLoadIdentity();

			camera.useView();

			for (int i = 0; i < ents.length; ++i) {
				ents[i].draw();
			}

			ps.update(delta);

			GL11.glBegin(GL11.GL_LINES);

			GL11.glColor3f(1.0f, 0.0f, 0.0f);
			GL11.glVertex3f(0.0f, 0.0f, 0.0f);
			GL11.glVertex3f(100.0f, 0.0f, 0.0f);

			GL11.glColor3f(0.0f, 1.0f, 0.0f);
			GL11.glVertex3f(0.0f, 0.0f, 0.0f);
			GL11.glVertex3f(0.0f, 100.0f, 0.0f);

			GL11.glColor3f(0.0f, 0.0f, 1.0f);
			GL11.glVertex3f(0.0f, 0.0f, 0.0f);
			GL11.glVertex3f(0.0f, 0.0f, 100.0f);

			GL11.glEnd();

			Display.update();
			Display.sync(FPS);
		}

		for (int i = 0; i < ents.length; ++i) {
			ents[i].cleanUp();
		}

		ps.delete();
		*/
	}

	public static void delete() {
		Display.destroy();
	}

	private static final int WIDTH = 800;
	private static final int HEIGHT = 600;
	private static final int FPS = 300;
	private static final String TITLE = "Компьютерная графика - лабораторная работа 4, 6, 7";
}
