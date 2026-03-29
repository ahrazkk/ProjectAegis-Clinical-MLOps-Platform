const fs = require('fs');
const filePath = 'src/components/GNNGalaxyViewer.jsx';
let code = fs.readFileSync(filePath, 'utf-8');

// Replace imports
code = code.replace(
  import { OrbitControls, Stars, Text, Line, Sphere, Trail } from '@react-three/drei';,
  import { OrbitControls, Stars, Text, Line, Sphere, Trail, Html } from '@react-three/drei';
);

// Fix the blinding white material
code = code.replace(
  const material = new THREE.MeshStandardMaterial({
    color: color,
    emissive: color,
    emissiveIntensity: 2,
    transparent: true,
    opacity: 0.8,
    metalness: 1,
    roughness: 0
  });,
  const material = new THREE.MeshStandardMaterial({
    color: color,
    emissive: color,
    emissiveIntensity: 0.5, // Toned down
    transparent: true,
    opacity: 0.1, // Almost glass-like
    wireframe: true, // Show skeletal structure
    metalness: 0.8,
    roughness: 0.2
  });
);

code = code.replace(
  <meshBasicMaterial color="#ffffff" />,
  <meshBasicMaterial color={color} />
);

// Update HexPoints code to add HTML explanations
let oldGroupReturn = 
  return (
    <group position={position} scale={scale} rotation={rotation} ref={group}>
      {/* Primary Hexagon - Extruded look via flattened cylinder */}
;

let newGroupReturn = 
  return (
    <group position={position} scale={scale} rotation={rotation} ref={group}>
      {/* Primary Hexagon - Extruded look via flattened cylinder */}
      {isComplex && (
        <Html position={[0, -2.5, 0]} center className="pointer-events-none">
          <div className="bg-black/80 border border-cyan-500/50 p-2 rounded text-xs text-cyan-200 shadow-[0_0_10px_rgba(0,255,255,0.2)] backdrop-blur-md whitespace-nowrap">
            <div className="font-bold text-[10px] text-cyan-400 mb-1 border-b border-cyan-800 pb-1">GraphSAGE Input Feature</div>
            SMILES Structural Embedding<br/>
            Latent Dimensions: 256
          </div>
        </Html>
      )}
;

code = code.replace(oldGroupReturn, newGroupReturn);

// Update the glowing center node size text
code = code.replace(/<mesh>\s*<sphereGeometry args=\{\[isComplex \? 0\.3 : 0\.15, 16, 16\]\} \/>\s*<meshBasicMaterial color=\{color\} \/>\s*<\/mesh>/g, 
  <mesh>
    <sphereGeometry args={[isComplex ? 0.3 : 0.15, 16, 16]} />
    <meshBasicMaterial color={color} />
  </mesh>
);


// Update GalaxyNodes to add explicit labels for GraphSAGE hops
let oldGalaxyNodes = eturn (
    <group>
      {/* Drug A Cluster */}
      <group>
        <SkeletalMolecule position={posA} color="#60a5fa" scale={2.5} isComplex={true} />
        <Text position={[-6, 3, 0]} fontSize={0.8} color="#93c5fd" anchorX="center" anchorY="middle">
          {drugs[0] || 'Drug A'}
        </Text>
;

let newGalaxyNodes = eturn (
    <group>
      {/* Drug A Cluster */}
      <group>
        <SkeletalMolecule position={posA} color="#60a5fa" scale={2.5} isComplex={true} />
        <Text position={[-6, 4, 0]} fontSize={0.8} color="#93c5fd" anchorX="center" anchorY="middle">
          {drugs[0] || 'Drug A'}
        </Text>
        <Html position={[-6, 3.2, 0]} center>
          <div className="text-[10px] text-blue-300/80 uppercase tracking-widest font-mono">Node A Representation</div>
        </Html>
;

code = code.replace(oldGalaxyNodes, newGalaxyNodes);

let oldDrugB = <SkeletalMolecule position={posB} color="#fb923c" scale={2.5} rotation={[0.5, 0.5, 0]} isComplex={true} />
        <Text position={[6, 3, 0]} fontSize={0.8} color="#fdba74" anchorX="center" anchorY="middle">
          {drugs[1] || 'Drug B'}
        </Text>;

let newDrugB = <SkeletalMolecule position={posB} color="#fb923c" scale={2.5} rotation={[0.5, 0.5, 0]} isComplex={true} />
        <Text position={[6, 4, 0]} fontSize={0.8} color="#fdba74" anchorX="center" anchorY="middle">
          {drugs[1] || 'Drug B'}
        </Text>
        <Html position={[6, 3.2, 0]} center>
          <div className="text-[10px] text-orange-300/80 uppercase tracking-widest font-mono">Node B Representation</div>
        </Html>;

code = code.replace(oldDrugB, newDrugB);

// The connective path explanation:
let oldPath = <Line points={[posA, middlePoint]} color={pathColor} lineWidth={2} dashed dashScale={20} dashSize={1} dashOffset={time} />
          <Line points={[middlePoint, posB]} color={pathColor} lineWidth={2} dashed dashScale={20} dashSize={1} dashOffset={-time} />;

let newPath = <Line points={[posA, middlePoint]} color={pathColor} lineWidth={2} dashed dashScale={20} dashSize={1} dashOffset={time} />
          <Line points={[middlePoint, posB]} color={pathColor} lineWidth={2} dashed dashScale={20} dashSize={1} dashOffset={-time} />
          
          <Html position={[middlePoint[0], middlePoint[1] + 1.5, middlePoint[2]]} center>
            <div className={\px-3 py-1.5 rounded-full border \ backdrop-blur-md shadow-lg flex flex-col items-center min-w-[200px]\}>
              <span className="text-[10px] uppercase font-bold tracking-widest mb-1 opacity-80">Latent Interaction Check</span>
              <span className="text-xs">{result ? \Dot Product Similarity: \\ : 'Calculating Similarity...'}</span>
              <div className="flex gap-2 mt-1">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                <span className="w-2 h-2 rounded-full bg-orange-500 animate-pulse delay-75"></span>
              </div>
            </div>
          </Html>
;

code = code.replace(oldPath, newPath);

// Add Hop node descriptions
let oldHopNode = const HopNode = ({ position, size, color, name }) => {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.5} />
      </mesh>
      <SkeletalMolecule position={[0,0,0]} scale={size * 1.5} color={color} isComplex={false} />
    </group>
  );
};;

let newHopNode = const HopNode = ({ position, size, color, name, isHub=false }) => {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1} wireframe={true} />
      </mesh>
      <SkeletalMolecule position={[0,0,0]} scale={size * 1.5} color={color} isComplex={false} />
      {isHub && (
        <Html position={[0, -0.6, 0]} center>
          <div className="bg-black/60 border border-gray-600/30 px-2 py-1 rounded text-[8px] text-gray-300 whitespace-nowrap">
            \
          </div>
        </Html>
      )}
    </group>
  );
};;

code = code.replace(oldHopNode, newHopNode);

code = code.replace(/<HopNode key=\{(.+?)\} position=\{pos\} size=\{0\.3\} color=\{pathColor\} \/>/g, 
  <HopNode key={} position={pos} size={0.3} color={pathColor} isHub={i===0} name={"Latent Hub"} />);

code = code.replace(/<HopNode key=\{(.+?)\} position=\{pos\} size=\{0\.15\} color="#8b5cf6" \/>/g, 
  <HopNode key={} position={pos} size={0.15} color="#8b5cf6" isHub={i===2} name={"1-Hop Neighbor"} />);

code = code.replace(/<HopNode key=\{(.+?)\} position=\{pos\} size=\{0\.1\} color="#3b82f6" \/>/g, 
  <HopNode key={} position={pos} size={0.1} color="#3b82f6" isHub={i===3} name={"2-Hop Neighbor"} />);

code = code.replace(/<HopNode key=\{(.+?)\} position=\{pos\} size=\{0\.15\} color="#fb7185" \/>/g, 
  <HopNode key={} position={pos} size={0.15} color="#fb7185" isHub={i===2} name={"1-Hop Neighbor"} />);

code = code.replace(/<HopNode key=\{(.+?)\} position=\{pos\} size=\{0\.1\} color="#f97316" \/>/g, 
  <HopNode key={} position={pos} size={0.1} color="#f97316" isHub={i===3} name={"2-Hop Neighbor"} />);

code = code.replace(/Bloom intensity=\{2\.5\}/, "Bloom intensity={1.2}");

fs.writeFileSync(filePath, code);
console.log('Done replacing visually blinding factors and adding tooltips.');
