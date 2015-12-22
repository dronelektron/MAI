package math;

public class Angle {
	public static Vector toVector(float pitch, float yaw) {
		float pitchRad = (float)Math.toRadians(pitch);
		float yawRad = (float)Math.toRadians(yaw);
		float x = (float)Math.cos(yawRad) * (float)Math.cos(pitchRad);
		float z = -(float)Math.sin(yawRad) * (float)Math.cos(pitchRad);
		float y = -(float)Math.sin(pitchRad);

		return new Vector(x, y, z, 0.0f);
	}
}
