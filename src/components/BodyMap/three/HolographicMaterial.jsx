/**
 * HolographicMaterial — Translucent holographic shader for React Three Fiber
 * Based on Anderson Mancini's holographic material (MIT-compatible, Dec 2023)
 * Produces the translucent blue wireframe + scanline + fresnel glow effect
 * seen in medical hologram UIs.
 */

import React, { useRef, useMemo } from 'react';
import { shaderMaterial } from '@react-three/drei';
import { extend, useFrame } from '@react-three/fiber';
import {
  Color,
  FrontSide,
  BackSide,
  DoubleSide,
  AdditiveBlending,
  NormalBlending,
} from 'three';

export default function HolographicMaterial({
  fresnelAmount = 0.45,
  fresnelOpacity = 1.0,
  scanlineSize = 8.0,
  hologramBrightness = 1.2,
  signalSpeed = 0.45,
  hologramColor = '#51a4de',
  enableBlinking = true,
  blinkFresnelOnly = true,
  enableAdditive = true,
  hologramOpacity = 1.0,
  side = 'FrontSide',
}) {
  const HolographicMat = useMemo(() => {
    return shaderMaterial(
      {
        time: 0,
        fresnelOpacity,
        fresnelAmount,
        scanlineSize,
        hologramBrightness,
        signalSpeed,
        hologramColor: new Color(hologramColor),
        enableBlinking,
        blinkFresnelOnly,
        hologramOpacity,
      },
      // Vertex shader
      /* glsl */ `
        #define STANDARD
        varying vec3 vViewPosition;
        varying vec2 vUv;
        varying vec4 vPos;
        varying vec3 vNormalW;
        varying vec3 vPositionW;

        #include <common>
        #include <uv_pars_vertex>
        #include <color_pars_vertex>
        #include <fog_pars_vertex>
        #include <morphtarget_pars_vertex>
        #include <skinning_pars_vertex>
        #include <logdepthbuf_pars_vertex>
        #include <clipping_planes_pars_vertex>

        void main() {
          #include <uv_vertex>
          #include <color_vertex>
          #include <morphcolor_vertex>
          #include <beginnormal_vertex>
          #include <morphnormal_vertex>
          #include <skinbase_vertex>
          #include <skinnormal_vertex>
          #include <defaultnormal_vertex>
          #include <begin_vertex>
          #include <morphtarget_vertex>
          #include <skinning_vertex>
          #include <project_vertex>
          #include <logdepthbuf_vertex>
          #include <clipping_planes_vertex>
          #include <worldpos_vertex>
          #include <fog_vertex>

          mat4 modelViewProjectionMatrix = projectionMatrix * modelViewMatrix;
          vUv = uv;
          vPos = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
          vPositionW = vec3(vec4(transformed, 1.0) * modelMatrix);
          vNormalW = normalize(vec3(vec4(normal, 0.0) * modelMatrix));
          gl_Position = modelViewProjectionMatrix * vec4(transformed, 1.0);
        }
      `,
      // Fragment shader
      /* glsl */ `
        varying vec2 vUv;
        varying vec3 vPositionW;
        varying vec4 vPos;
        varying vec3 vNormalW;

        uniform float time;
        uniform float fresnelOpacity;
        uniform float scanlineSize;
        uniform float fresnelAmount;
        uniform float signalSpeed;
        uniform float hologramBrightness;
        uniform float hologramOpacity;
        uniform bool blinkFresnelOnly;
        uniform bool enableBlinking;
        uniform vec3 hologramColor;

        float flicker(float amt, float time) {
          return clamp(fract(cos(time) * 43758.5453123), amt, 1.0);
        }

        float random(in float a, in float b) {
          return fract((cos(dot(vec2(a, b), vec2(12.9898, 78.233))) * 43758.5453));
        }

        void main() {
          vec2 vCoords = vPos.xy;
          vCoords /= vPos.w;
          vCoords = vCoords * 0.5 + 0.5;
          vec2 myUV = fract(vCoords);

          vec4 hColor = vec4(hologramColor, mix(hologramBrightness, vUv.y, 0.5));

          // Scanlines
          float scanlines = 10.0;
          scanlines += 20.0 * sin(time * signalSpeed * 20.8 - myUV.y * 60.0 * scanlineSize);
          scanlines *= smoothstep(1.3 * cos(time * signalSpeed + myUV.y * scanlineSize), 0.78, 0.9);
          scanlines *= max(0.25, sin(time * signalSpeed) * 1.0);

          float r = random(vUv.x, vUv.y);
          float g = random(vUv.y * 20.2, vUv.y * 0.2);
          float b = random(vUv.y * 0.9, vUv.y * 0.2);

          hColor += vec4(r * scanlines, b * scanlines, r, 1.0) / 84.0;
          vec4 scanlineMix = mix(vec4(0.0), hColor, hColor.a);

          // Fresnel
          vec3 viewDirectionW = normalize(cameraPosition - vPositionW);
          float fresnelEffect = dot(viewDirectionW, vNormalW) * (1.6 - fresnelOpacity / 2.0);
          fresnelEffect = clamp(fresnelAmount - fresnelEffect, 0.0, fresnelOpacity);

          // Blink
          float blinkValue = enableBlinking ? 0.6 - signalSpeed : 1.0;
          float blink = flicker(blinkValue, time * signalSpeed * 0.02);

          vec3 finalColor;
          if (blinkFresnelOnly) {
            finalColor = scanlineMix.rgb + fresnelEffect * blink;
          } else {
            finalColor = scanlineMix.rgb * blink + fresnelEffect;
          }

          gl_FragColor = vec4(finalColor, hologramOpacity);
        }
      `
    );
  }, [
    fresnelAmount, fresnelOpacity, scanlineSize, hologramBrightness,
    signalSpeed, hologramColor, enableBlinking, blinkFresnelOnly,
    enableAdditive, hologramOpacity, side,
  ]);

  extend({ HolographicMat });

  const ref = useRef();

  useFrame((_, delta) => {
    if (ref.current) ref.current.time += delta;
  });

  const sideEnum = side === 'DoubleSide' ? DoubleSide : side === 'BackSide' ? BackSide : FrontSide;

  return (
    <holographicMat
      key={HolographicMat.key}
      side={sideEnum}
      transparent={true}
      blending={enableAdditive ? AdditiveBlending : NormalBlending}
      depthWrite={false}
      ref={ref}
    />
  );
}
