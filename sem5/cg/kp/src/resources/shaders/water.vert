#version 130

uniform mat4 u_viewProj;
uniform mat4 u_model;
uniform vec4 u_cameraPos;
uniform vec4 u_lightPos;

varying vec4 v_clipSpace;
varying vec2 v_texCoords;
varying vec3 v_toCameraVector;
varying vec3 v_fromLightVector;

float tiling = 24.0;

void main() {
	vec4 worldPos = u_model * gl_Vertex;
	
	v_clipSpace = u_viewProj * worldPos;
	v_texCoords = vec2(gl_MultiTexCoord0.x / 2.0 + 0.5, gl_MultiTexCoord0.y / 2.0 + 0.5) * tiling;
	v_toCameraVector = u_cameraPos.xyz - worldPos.xyz;
	v_fromLightVector = worldPos.xyz - u_lightPos.xyz;
	gl_Position = v_clipSpace;
}
