package main;

import java.nio.ByteBuffer;
import org.lwjgl.opengl.*;

public class WaterFBO {
	public WaterFBO() {
		initialiseReflectionFrameBuffer();
		initialiseRefractionFrameBuffer();
	}

	public void delete() {
		GL30.glDeleteFramebuffers(reflectionFrameBuffer);
		GL11.glDeleteTextures(reflectionTexture);
		GL30.glDeleteRenderbuffers(reflectionDepthBuffer);
		GL30.glDeleteFramebuffers(refractionFrameBuffer);
		GL11.glDeleteTextures(refractionTexture);
		GL11.glDeleteTextures(refractionDepthTexture);
	}

	public void bindReflectionFrameBuffer() {
		bindFrameBuffer(reflectionFrameBuffer, Display.getWidth(), Display.getHeight());
	}

	public void bindRefractionFrameBuffer() {
		bindFrameBuffer(refractionFrameBuffer, Display.getWidth(), Display.getHeight());
	}

	public void unbindCurrentFrameBuffer() {
		GL30.glBindFramebuffer(GL30.GL_FRAMEBUFFER, 0);
		GL11.glViewport(0, 0, Display.getWidth(), Display.getHeight());
	}

	public int getReflectionTexture() {
		return reflectionTexture;
	}

	public int getRefractionTexture() {
		return refractionTexture;
	}

	public int getRefractionDepthTexture() {
		return refractionDepthTexture;
	}

	private void initialiseReflectionFrameBuffer() {
		reflectionFrameBuffer = createFrameBuffer();
		reflectionTexture = createTextureAttachment(Display.getWidth(), Display.getHeight());
		reflectionDepthBuffer = createDepthBufferAttachment(Display.getWidth(), Display.getHeight());
		unbindCurrentFrameBuffer();
	}

	private void initialiseRefractionFrameBuffer() {
		refractionFrameBuffer = createFrameBuffer();
		refractionTexture = createTextureAttachment(Display.getWidth(), Display.getHeight());
		refractionDepthTexture = createDepthTextureAttachment(Display.getWidth(), Display.getHeight());
		unbindCurrentFrameBuffer();
	}

	private void bindFrameBuffer(int frameBuffer, int width, int height) {
		GL11.glBindTexture(GL11.GL_TEXTURE_2D, 0);
		GL30.glBindFramebuffer(GL30.GL_FRAMEBUFFER, frameBuffer);
		GL11.glViewport(0, 0, width, height);
	}

	private int createFrameBuffer() {
		int frameBuffer = GL30.glGenFramebuffers();

		GL30.glBindFramebuffer(GL30.GL_FRAMEBUFFER, frameBuffer);
		GL11.glDrawBuffer(GL30.GL_COLOR_ATTACHMENT0);

		return frameBuffer;
	}

	private int createTextureAttachment(int width, int height) {
		int texture = GL11.glGenTextures();

		GL11.glBindTexture(GL11.GL_TEXTURE_2D, texture);
		GL11.glTexImage2D(GL11.GL_TEXTURE_2D, 0, GL11.GL_RGB, width, height, 0, GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, (ByteBuffer)null);
		GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_LINEAR);
		GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR);
		GL32.glFramebufferTexture(GL30.GL_FRAMEBUFFER, GL30.GL_COLOR_ATTACHMENT0, texture, 0);

		return texture;
	}

	private int createDepthTextureAttachment(int width, int height) {
		int texture = GL11.glGenTextures();

		GL11.glBindTexture(GL11.GL_TEXTURE_2D, texture);
		GL11.glTexImage2D(GL11.GL_TEXTURE_2D, 0, GL14.GL_DEPTH_COMPONENT32, width, height, 0, GL11.GL_DEPTH_COMPONENT, GL11.GL_FLOAT, (ByteBuffer)null);
		GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_LINEAR);
		GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR);
		GL32.glFramebufferTexture(GL30.GL_FRAMEBUFFER, GL30.GL_DEPTH_ATTACHMENT, texture, 0);

		return texture;
	}

	private int createDepthBufferAttachment(int width, int height) {
		int buffer = GL30.glGenRenderbuffers();

		GL30.glBindRenderbuffer(GL30.GL_RENDERBUFFER, buffer);
		GL30.glRenderbufferStorage(GL30.GL_RENDERBUFFER, GL11.GL_DEPTH_COMPONENT, width, height);
		GL30.glFramebufferRenderbuffer(GL30.GL_FRAMEBUFFER, GL30.GL_DEPTH_ATTACHMENT, GL30.GL_RENDERBUFFER, buffer);

		return buffer;
	}

	private int reflectionFrameBuffer;
	private int reflectionTexture;
	private int reflectionDepthBuffer;
	private int refractionFrameBuffer;
	private int refractionTexture;
	private int refractionDepthTexture;
}
