#version 130

uniform mat4 u_viewProj;
uniform vec4 u_plane;

varying vec2 v_texCoord;

void main() {
	gl_ClipDistance[0] = dot(gl_Vertex, u_plane);
	gl_Position = u_viewProj * gl_Vertex;
	v_texCoord = gl_MultiTexCoord0.xy;
}
