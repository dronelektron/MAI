package application;

import objects.*;
import org.lwjgl.LWJGLException;
import org.lwjgl.input.Keyboard;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;
import org.lwjgl.opengl.GL11;

public class Main {
	public static void main(String[] args) {
		initDisplay();
		initGL();
		loop();
		cleanUp();
	}

	public static void initDisplay() {
		final int WIDTH = 800;
		final int HEIGHT = 600;
		final String title = "Компьютерная графика - лабораторная работа 4";

		try {
			Display.setDisplayMode(new DisplayMode(WIDTH, HEIGHT));
			Display.setTitle(title);
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
	}

	public static void loop() {
		Camera camera = new Camera(Display.getWidth(), Display.getHeight());
		Entity[] ents = new Entity[] {new Terrain(), new Tree()};

		for (int i = 0; i < ents.length; ++i) {
			ents[i].compile();
		}

		camera.setPos(256.0f, 256.0f, 256.0f);

		long prevTime = System.nanoTime();
		//long frames = 0;
		//float time = 0.0f;

		while (!Display.isCloseRequested()) {
			long curTime = System.nanoTime();
			float delta = (curTime - prevTime) / 1000000000.0f;

			prevTime = curTime;
			//time += delta;

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

			// ОСИ
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
			Display.sync(60);
			/*
			++frames;

			if (time >= 1.0f) {
				System.out.println("FPS: " + frames);

				frames = 0;
				time = 0.0f;
			}
			*/
		}

		for (int i = 0; i < ents.length; ++i) {
			ents[i].cleanUp();
		}
	}

	public static void cleanUp() {
		Display.destroy();
	}
}
