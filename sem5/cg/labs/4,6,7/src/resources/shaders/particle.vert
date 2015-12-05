#version 120

uniform mat4 u_mvp;
uniform float u_lifetime;
uniform float u_maxlifetime;

varying vec2 v_texCoord;
varying float v_alpha;

void main() {
	float t = u_lifetime / u_maxlifetime;

	gl_Position = u_mvp * gl_Vertex;
	v_texCoord = gl_MultiTexCoord0.xy;
	v_alpha = 1.0 - t;
}
