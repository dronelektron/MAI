package objects;

import org.lwjgl.opengl.GL11;
import math.Matrix;
import math.Vector;

public abstract class Entity {
	public Entity() {
		dispList = GL11.glGenLists(1);
	}

	public abstract void compile();
	public abstract void draw(Matrix projMat, Matrix viewMat, Vector plane, Vector cameraPos);

	public void update(float delta) {}

	public void delete() {
		GL11.glDeleteLists(dispList, 1);
	}

	protected int dispList;
}
