#version 130

uniform mat4 u_viewProj;
uniform mat4 u_model;
uniform vec4 u_plane;

varying vec2 v_texCoord;

void main() {
	vec4 pos = u_model * gl_Vertex;

	gl_ClipDistance[0] = dot(pos, u_plane);
	gl_Position = u_viewProj * pos;
	v_texCoord = gl_MultiTexCoord0.xy;
}
