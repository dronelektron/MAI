#version 130

uniform sampler2D u_reflectionTexture;
uniform sampler2D u_refractionTexture;
uniform sampler2D u_dudvMap;
uniform sampler2D u_normalMap;
uniform float u_moveFactor;

varying vec4 v_clipSpace;
varying vec2 v_texCoords;
varying vec3 v_toCameraVector;
varying vec3 v_fromLightVector;

vec3 sunColor = vec3(0.8, 0.6, 0.0);
float waveStrength = 0.02;
float shineDamper = 20.0;
float reflectivity = 0.6;
float specPow = 2.0;

void main() {
	vec2 ndc = (v_clipSpace.xy / v_clipSpace.w) / 2.0 + 0.5;
	vec2 refractTexCoords = vec2(ndc.x, ndc.y);
	vec2 reflectTexCoords = vec2(ndc.x, -ndc.y);

	vec2 distortedTexCoords = texture(u_dudvMap, vec2(v_texCoords.x + u_moveFactor, v_texCoords.y)).rg * 0.1;
	distortedTexCoords = v_texCoords + vec2(distortedTexCoords.x, distortedTexCoords.y + u_moveFactor);
	vec2 totalDistortion = (texture(u_dudvMap, distortedTexCoords).rg * 2.0 - 1.0) * waveStrength;

	refractTexCoords += totalDistortion;
	refractTexCoords = clamp(refractTexCoords, 0.001, 0.999);

	reflectTexCoords += totalDistortion;
	reflectTexCoords.x = clamp(reflectTexCoords.x, 0.001, 0.999);
	reflectTexCoords.y = clamp(reflectTexCoords.y, -0.999, -0.001);

	vec4 reflectColor = texture(u_reflectionTexture, reflectTexCoords);
	vec4 refractColor = texture(u_refractionTexture, refractTexCoords);

	vec3 viewVector = normalize(v_toCameraVector);
	float refractiveFactor = dot(viewVector, vec3(0.0, 1.0, 0.0));

	refractiveFactor = pow(refractiveFactor, specPow);

	vec4 normalMapColor = texture(u_normalMap, distortedTexCoords);
	vec3 normal = vec3(normalMapColor.r * 2.0 - 1.0, normalMapColor.b, normalMapColor.g * 2.0 - 1.0);

	normal = normalize(normal);

	vec3 reflectedLight = reflect(normalize(v_fromLightVector), normal);
	float specular = max(dot(reflectedLight, viewVector), 0.0);
	specular = pow(specular, shineDamper);
	vec3 specularHL = sunColor * specular * reflectivity;

	gl_FragColor = mix(reflectColor, refractColor, refractiveFactor);
	gl_FragColor = mix(gl_FragColor, vec4(0.0, 0.3, 0.5, 1.0), 0.2) + vec4(specularHL, 0.0);
}
