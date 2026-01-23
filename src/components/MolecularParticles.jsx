import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree, createPortal } from '@react-three/fiber';
import { useFBO } from '@react-three/drei';
import * as THREE from 'three';

// DNA-like helix structure with molecular nodes
function MolecularHelix({ scrollProgress }) {
  const groupRef = useRef();
  const particleCount = 300;

  const { positions, colors, sizes } = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      const t = (i / particleCount) * Math.PI * 8;
      const strand = i % 2 === 0 ? 1 : -1;
      const radius = 3 + Math.sin(t * 0.5) * 0.5;

      positions[i * 3] = Math.cos(t) * radius * strand;
      positions[i * 3 + 1] = (i / particleCount - 0.5) * 15;
      positions[i * 3 + 2] = Math.sin(t) * radius;

      const colorT = i / particleCount;
      colors[i * 3] = 0.2 + colorT * 0.2;
      colors[i * 3 + 1] = 0.5 + colorT * 0.3;
      colors[i * 3 + 2] = 0.9 + Math.random() * 0.1;

      sizes[i] = 0.05 + Math.random() * 0.08;
    }

    return { positions, colors, sizes };
  }, []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
      groupRef.current.rotation.x = scrollProgress * Math.PI * 0.15;
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.5;
    }
  });

  return (
    <group ref={groupRef}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={particleCount} array={positions} itemSize={3} />
          <bufferAttribute attach="attributes-color" count={particleCount} array={colors} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial
          size={0.12}
          vertexColors
          transparent
          opacity={0.9}
          sizeAttenuation
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>
    </group>
  );
}

// Floating molecular nodes
function FloatingNodes() {
  const groupRef = useRef();
  const nodeCount = 50;

  const nodes = useMemo(() => {
    return Array.from({ length: nodeCount }, (_, i) => ({
      position: [
        (Math.random() - 0.5) * 25,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20
      ],
      size: 0.1 + Math.random() * 0.15,
      speed: 0.3 + Math.random() * 0.5,
      offset: Math.random() * Math.PI * 2
    }));
  }, []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <group ref={groupRef}>
      {nodes.map((node, i) => (
        <mesh key={i} position={node.position}>
          <sphereGeometry args={[node.size, 8, 8]} />
          <meshBasicMaterial
            color={i % 3 === 0 ? '#3B82F6' : i % 3 === 1 ? '#06B6D4' : '#8B5CF6'}
            transparent
            opacity={0.4}
          />
        </mesh>
      ))}
    </group>
  );
}

// Background particle cloud
function ParticleCloud({ scrollProgress }) {
  const pointsRef = useRef();
  const particleCount = 800;

  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = 8 + Math.random() * 12;

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);

      const brightness = 0.3 + Math.random() * 0.4;
      colors[i * 3] = brightness * 0.3;
      colors[i * 3 + 1] = brightness * 0.5;
      colors[i * 3 + 2] = brightness * 1.0;
    }

    return { positions, colors };
  }, []);

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.03;
      pointsRef.current.rotation.x = scrollProgress * Math.PI * 0.1;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={particleCount} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={particleCount} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.04}
        vertexColors
        transparent
        opacity={0.6}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// The actual 3D scene content
function SceneContent({ scrollProgress }) {
  return (
    <group>
      <ParticleCloud scrollProgress={scrollProgress} />
      <MolecularHelix scrollProgress={scrollProgress} />
      <FloatingNodes />
      <ambientLight intensity={0.3} />
    </group>
  );
}

// Fractured glass distortion shader
const distortionVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const distortionFragmentShader = `
  uniform sampler2D tDiffuse;
  uniform vec2 uMouse;
  uniform float uRadius;
  uniform float uTime;
  uniform float uActive;
  varying vec2 vUv;

  void main() {
    vec2 uv = vUv;
    vec4 color = texture2D(tDiffuse, uv);
    
    if (uActive > 0.5) {
      vec2 diff = uv - uMouse;
      float dist = length(diff);
      
      if (dist < uRadius) {
        float normalizedDist = dist / uRadius;
        float falloff = 1.0 - smoothstep(0.0, 1.0, normalizedDist);
        
        // Create shard-like distortion pattern
        float angle = atan(diff.y, diff.x);
        float shards = 12.0;
        float shardIndex = floor((angle + 3.14159) / (6.28318 / shards));
        
        // Each shard has different displacement
        float shardOffset = sin(shardIndex * 2.1 + uTime * 3.0) * 0.5 + 0.5;
        float shardRotation = cos(shardIndex * 1.7) * 0.4;
        
        // Calculate displacement
        vec2 shardDir = vec2(cos(angle + shardRotation), sin(angle + shardRotation));
        float displacement = falloff * (0.04 + shardOffset * 0.03);
        
        // Apply distortion
        vec2 distortedUV = uv + shardDir * displacement;
        distortedUV = clamp(distortedUV, 0.0, 1.0);
        
        // Sample with chromatic aberration
        float chromaOffset = falloff * 0.008;
        float r = texture2D(tDiffuse, distortedUV + vec2(chromaOffset, 0.0)).r;
        float g = texture2D(tDiffuse, distortedUV).g;
        float b = texture2D(tDiffuse, distortedUV - vec2(chromaOffset, 0.0)).b;
        
        color = vec4(r, g, b, 1.0);
        
        // Add glow at shard edges
        float edgeDist = abs(mod(angle + 3.14159, 6.28318 / shards) - 3.14159 / shards);
        float edgeGlow = smoothstep(0.15, 0.0, edgeDist) * falloff * 0.4;
        color.rgb += vec3(0.0, 0.8, 1.0) * edgeGlow;
        
        // Add center impact glow
        float impactGlow = smoothstep(0.2, 0.0, normalizedDist) * 0.3;
        color.rgb += vec3(0.0, 0.9, 1.0) * impactGlow;
      }
    }
    
    gl_FragColor = color;
  }
`;

// Fullscreen distortion plane that renders the scene with effects
function DistortionPlane({ scrollProgress }) {
  const { gl, size, camera, scene } = useThree();
  const meshRef = useRef();
  const sceneRef = useRef(new THREE.Scene());
  const mouseRef = useRef({ x: 0.5, y: 0.5, active: false });
  
  // Create render target
  const renderTarget = useFBO(size.width, size.height, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
  });

  // Create shader material
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null },
        uMouse: { value: new THREE.Vector2(0.5, 0.5) },
        uRadius: { value: 0.15 },
        uTime: { value: 0 },
        uActive: { value: 0 },
      },
      vertexShader: distortionVertexShader,
      fragmentShader: distortionFragmentShader,
      transparent: true,
    });
  }, []);

  // Mouse tracking
  useEffect(() => {
    const handleMouseMove = (e) => {
      mouseRef.current.x = e.clientX / size.width;
      mouseRef.current.y = 1.0 - (e.clientY / size.height);
      mouseRef.current.active = true;
    };
    const handleMouseLeave = () => {
      mouseRef.current.active = false;
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [size]);

  useFrame((state) => {
    if (!meshRef.current) return;

    // Update uniforms
    material.uniforms.uTime.value = state.clock.elapsedTime;
    material.uniforms.uMouse.value.set(mouseRef.current.x, mouseRef.current.y);
    material.uniforms.uActive.value = mouseRef.current.active ? 1.0 : 0.0;
    
    // Render scene to texture
    gl.setRenderTarget(renderTarget);
    gl.render(sceneRef.current, camera);
    gl.setRenderTarget(null);
    
    // Update texture
    material.uniforms.tDiffuse.value = renderTarget.texture;
  });

  return (
    <>
      {/* Portal renders the 3D content into our separate scene */}
      {createPortal(<SceneContent scrollProgress={scrollProgress} />, sceneRef.current)}
      
      {/* Fullscreen quad with distortion shader */}
      <mesh ref={meshRef} frustumCulled={false}>
        <planeGeometry args={[2, 2]} />
        <primitive object={material} attach="material" />
      </mesh>
    </>
  );
}

export default function ParticleSystem({ scrollProgress = 0 }) {
  return <DistortionPlane scrollProgress={scrollProgress} />;
}
