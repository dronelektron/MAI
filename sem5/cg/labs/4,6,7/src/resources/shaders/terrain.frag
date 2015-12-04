#version 120

uniform sampler2D u_sampler;

varying vec2 v_texCoord;

void main() {
	vec4 color = texture2D(u_sampler, v_texCoord);

    gl_FragColor = color;
}
