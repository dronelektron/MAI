#version 120

uniform sampler2D u_sampler;

varying vec2 v_texCoord;
varying float v_alpha;

void main() {
	vec4 texcolor = texture2D(u_sampler, v_texCoord);
	gl_FragColor = vec4(texcolor.xyz, texcolor.a * v_alpha);
}
