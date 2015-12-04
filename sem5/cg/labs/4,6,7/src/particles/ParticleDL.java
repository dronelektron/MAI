package particles;

import org.lwjgl.opengl.GL11;

public class ParticleDL {
	public ParticleDL() {
		displayList = GL11.glGenLists(1);

		GL11.glNewList(displayList, GL11.GL_COMPILE);
		GL11.glBegin(GL11.GL_QUADS);
		GL11.glTexCoord2f(0.0f, 0.0f);
		GL11.glVertex3f(-0.5f, 0.5f, 0.0f);
		GL11.glTexCoord2f(0.0f, 1.0f);
		GL11.glVertex3f(-0.5f, -0.5f, 0.0f);
		GL11.glTexCoord2f(1.0f, 1.0f);
		GL11.glVertex3f(0.5f, -0.5f, 0.0f);
		GL11.glTexCoord2f(1.0f, 0.0f);
		GL11.glVertex3f(0.5f, 0.5f, 0.0f);
		GL11.glEnd();
		GL11.glEndList();
	}

	public void draw() {
		GL11.glCallList(displayList);
	}

	public void delete() {
		GL11.glDeleteLists(displayList, 1);
	}

	private int displayList;
}
