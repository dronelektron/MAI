#version 120

uniform mat4 u_mvp;
uniform vec4 u_velocity;
uniform float u_lifetime;
uniform float u_maxlifetime;

varying vec2 v_texCoord;
varying float v_alpha;

void main() {
	float t = u_lifetime / u_maxlifetime;
	vec3 delta = u_velocity.xyz * u_lifetime;
	vec3 pos = gl_Vertex.xyz + delta;

	gl_Position = u_mvp * vec4(pos, 1.0);
	v_texCoord = gl_MultiTexCoord0.xy;
	v_alpha = 1.0 - t;
}
