package main;

import objects.Entity;
import objects.Terrain;
import org.lwjgl.LWJGLException;
import org.lwjgl.input.Keyboard;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;
import org.lwjgl.opengl.GL11;
import math.Matrix;
import particles.ParticleSystem;

public class Main {
	public static void main(String[] args) {
		initDisplay();
		initGL();
		loop();
		delete();
	}

	private static void initDisplay() {
		try {
			final int WIDTH = 800;
			final int HEIGHT = 600;

			Display.setDisplayMode(new DisplayMode(WIDTH, HEIGHT));
			Display.setTitle(TITLE);
			Display.setResizable(true);
			Display.create();

			prevTime = 0;
			camera = new Camera(WIDTH, HEIGHT);
			projection = new Matrix().initPerspective(75.0f, (float)WIDTH / HEIGHT, 0.1f, 1000.0f);
			terrainShader = new Shader("src/resources/shaders/terrain");
			particleSystem = new ParticleSystem(50);
			terrainObj = new Terrain();
			terrainObj.compile();
		} catch (LWJGLException e) {
			e.printStackTrace();
		}
	}

	private static void initGL() {
		GL11.glEnable(GL11.GL_DEPTH_TEST);
		GL11.glEnable(GL11.GL_CULL_FACE);
		GL11.glEnable(GL11.GL_TEXTURE_2D);
		GL11.glEnable(GL11.GL_BLEND);
		GL11.glBlendFunc(GL11.GL_SRC_ALPHA, GL11.GL_ONE_MINUS_SRC_ALPHA);
	}

	private static void loop() {
		while (!Display.isCloseRequested()) {
			getInput();
			draw();

			Display.update();
			Display.sync(FPS);
		}
	}

	private static void delete() {
		particleSystem.delete();
		terrainShader.delete();
		terrainObj.delete();

		Display.destroy();
	}

	private static void draw() {
		//float delta = getDelta();
		Matrix view = camera.getView();
		Matrix viewProjMat = projection.mul(view);

		GL11.glClearColor(0.0f, 0.5f, 1.0f, 1.0f);
		GL11.glClear(GL11.GL_COLOR_BUFFER_BIT | GL11.GL_DEPTH_BUFFER_BIT);

		// TODO: Draw stuff
		terrainShader.bind();
		terrainShader.setUniform1("u_sampler", 0);
		terrainShader.setUniformMatrix4("u_mvp", viewProjMat.toBuffer());
		terrainObj.draw();

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

		terrainShader.unbind();
	}

	private static void getInput() {
		float delta = getDelta();

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

	private static float getDelta() {
		long curTime = System.nanoTime();
		float delta = (curTime - prevTime) / 1000000000.0f;

		prevTime = curTime;

		return delta;
	}

	private static final int FPS = 60;
	private static final String TITLE = "Компьютерная графика - лабораторная работа 4, 6, 7";

	private static long prevTime;
	private static Camera camera;
	private static Matrix projection;
	private static Shader terrainShader;
	private static ParticleSystem particleSystem;
	private static Entity terrainObj;
}
