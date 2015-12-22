#version 130

uniform mat4 u_viewProj;
uniform mat4 u_model;
uniform vec4 u_plane;
uniform float u_lifetime;
uniform float u_maxlifetime;

varying vec2 v_texCoord;
varying float v_alpha;

void main() {
	float t = u_lifetime / u_maxlifetime;
	vec4 pos = u_model * gl_Vertex;
	gl_ClipDistance[0] = dot(pos, u_plane);
    gl_Position = u_viewProj * pos;
	v_texCoord = gl_MultiTexCoord0.xy;
	v_alpha = 1.0 - t;
}
