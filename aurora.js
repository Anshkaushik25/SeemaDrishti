import { Renderer, Program, Mesh, Color, Triangle } from 'https://esm.sh/ogl@1.0.11';

const VERT = `#version 300 es
in vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
uniform float uTime;
uniform float uAmplitude;
uniform vec3 uColorStops[3];
uniform vec2 uResolution;
uniform float uBlend;
out vec4 fragColor;

vec3 permute(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m; m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

struct ColorStop {
  vec3 color;
  float position;
};

#define COLOR_RAMP(colors, factor, finalColor) { \
  int index = 0; \
  for (int i = 0; i < 5; i++) { \
    ColorStop c = colors[i]; \
    bool b = c.position <= factor; \
    index = int(mix(float(index), float(i), float(b))); \
  } \
  ColorStop c1 = colors[index]; \
  ColorStop c2 = colors[index + 1]; \
  float range = max(c2.position - c1.position, 0.0001); \
  float lf = clamp((factor - c1.position) / range, 0.0, 1.0); \
  finalColor = mix(c1.color, c2.color, lf); \
}

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution;

  // Exactly 3 Equal Thirds: Saffron (0.0 - 0.333) | White (0.333 - 0.666) | Green (0.666 - 1.0)
  ColorStop colors[6];
  colors[0] = ColorStop(uColorStops[0], 0.00);  // Saffron start
  colors[1] = ColorStop(uColorStops[0], 0.24);  // Saffron solid
  colors[2] = ColorStop(uColorStops[1], 0.42);  // White start
  colors[3] = ColorStop(uColorStops[1], 0.58);  // White end
  colors[4] = ColorStop(uColorStops[2], 0.76);  // Green solid
  colors[5] = ColorStop(uColorStops[2], 1.00);  // Green end

  vec3 rampColor;
  COLOR_RAMP(colors, uv.x, rampColor);

  // Full vertical projection reaching down past the lower viewport edge
  float noiseVal = snoise(vec2(uv.x * 1.6 + uTime * 0.08, uTime * 0.2)) * 0.45 * uAmplitude;
  float height = exp(noiseVal);
  height = (uv.y * 0.80 - height + 0.95);

  float intensity = 0.80 * height;

  float midPoint = -0.05;
  float auroraAlpha = smoothstep(midPoint - uBlend * 0.6, midPoint + uBlend * 0.6, intensity);
  vec3 auroraColor = intensity * rampColor;
  fragColor = vec4(auroraColor * auroraAlpha, auroraAlpha);
}
`;

const ctn = document.getElementById('aurora-bg');
const renderer = new Renderer({ alpha: true, premultipliedAlpha: true, antialias: true });
const gl = renderer.gl;
gl.clearColor(0, 0, 0, 0);
gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
gl.canvas.style.width = '100%';
gl.canvas.style.height = '100%';

let program;
function resize() {
  const w = ctn.offsetWidth, h = ctn.offsetHeight;
  renderer.setSize(w, h);
  if (program) program.uniforms.uResolution.value = [w, h];
}
window.addEventListener('resize', resize);

const colorStops = ['#FF9226', '#FFFFFF', '#138808'].map(hex => {
  const c = new Color(hex);
  return [c.r, c.g, c.b];
});

program = new Program(gl, {
  vertex: VERT,
  fragment: FRAG,
  uniforms: {
    uTime: { value: 0 },
    uAmplitude: { value: 1.4 },
    uColorStops: { value: colorStops },
    uResolution: { value: [ctn.offsetWidth, ctn.offsetHeight] },
    uBlend: { value: 0.7 }
  }
});

const mesh = new Mesh(gl, { geometry: new Triangle(gl), program });
ctn.appendChild(gl.canvas);

window.setAuroraState = (opacity, amplitude = 1.4, blend = 0.7, speed = 0.5) => {
  ctn.style.opacity = opacity;
  program.uniforms.uAmplitude.value = amplitude;
  program.uniforms.uBlend.value = blend;
  window.auroraSpeed = speed;
};

window.auroraSpeed = 0.5;
const update = t => {
  requestAnimationFrame(update);
  program.uniforms.uTime.value = (t * 0.01) * window.auroraSpeed * 0.1;
  renderer.render({ scene: mesh });
};
requestAnimationFrame(update);
resize();
