#version 120

uniform mat4 u_mvp;

varying vec2 v_texCoord;

void main() {
	gl_Position = u_mvp * gl_Vertex;
	v_texCoord = gl_MultiTexCoord0.xy;
}
