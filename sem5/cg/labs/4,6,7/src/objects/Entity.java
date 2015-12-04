package objects;

import org.lwjgl.opengl.GL11;
import java.util.ArrayList;

public class Entity {
	public Entity() {
		points = new ArrayList<>();
		texCoords = new ArrayList<>();
		indices = new ArrayList<>();
	}

	public void compile() {
		dispList = GL11.glGenLists(1);

		GL11.glNewList(dispList, GL11.GL_COMPILE);
		GL11.glBegin(GL11.GL_TRIANGLES);
		GL11.glColor3f(1.0f, 1.0f, 1.0f);

		for (int i = 0; i < indices.size(); i += 3) {
			for (int j = 0; j < 3; ++j) {
				GL11.glTexCoord2f(texCoords.get(2 * indices.get(i + j)),
						texCoords.get(2 * indices.get(i + j) + 1));

				GL11.glVertex3f(
						points.get(3 * indices.get(i + j)),
						points.get(3 * indices.get(i + j) + 1),
						points.get(3 * indices.get(i + j) + 2));
			}
		}

		GL11.glEnd();
		GL11.glEndList();
	}

	public void draw() {
		GL11.glCallList(dispList);
	}

	public void delete() {
		GL11.glDeleteLists(dispList, 1);
	}

	protected ArrayList<Float> points;
	protected ArrayList<Float> texCoords;
	protected ArrayList<Integer> indices;
	private int dispList;
}
