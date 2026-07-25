'use client';

import React, { useEffect, useRef } from 'react';
import { NetworkData, NetworkNode, NetworkEdge } from '@/lib/api';
import * as THREE from 'three';

interface NetworkGraph3DProps {
  data: NetworkData;
  onNodeClick?: (accountId: string) => void;
}

export const NetworkGraph3D: React.FC<NetworkGraph3DProps> = ({ data, onNodeClick }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const isDestroyed = useRef(false);

  useEffect(() => {
    if (!containerRef.current || !data.nodes.length) return;
    isDestroyed.current = false;

    const width = containerRef.current.clientWidth || 500;
    const height = containerRef.current.clientHeight || 500;

    // Create Scene
    const scene = new THREE.Scene();

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 200;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);

    // Group for rotation
    const graphGroup = new THREE.Group();
    scene.add(graphGroup);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 0.8, 500);
    pointLight.position.set(100, 100, 100);
    scene.add(pointLight);

    // Node Positions map (random 3D layout inside a sphere)
    const nodePositions: Record<string, THREE.Vector3> = {};
    const radius = 90;

    data.nodes.forEach((node) => {
      // Generate spherical coordinates
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      
      const factor = 0.3 + 0.7 * Math.random(); // spread nodes inwards and outwards
      const x = radius * Math.sin(phi) * Math.cos(theta) * factor;
      const y = radius * Math.sin(phi) * Math.sin(theta) * factor;
      const z = radius * Math.cos(phi) * factor;
      
      nodePositions[node.id] = new THREE.Vector3(x, y, z);
    });

    // Create Edges (Lines)
    const edgesGroup = new THREE.Group();
    graphGroup.add(edgesGroup);

    data.edges.forEach((edge) => {
      const pSource = nodePositions[edge.source];
      const pTarget = nodePositions[edge.target];

      if (pSource && pTarget) {
        const edgeGeometry = new THREE.BufferGeometry().setFromPoints([pSource, pTarget]);
        
        // Simple neon white-blue wireframe edges
        const edgeMaterial = new THREE.LineBasicMaterial({
          color: 0x475569,
          transparent: true,
          opacity: 0.3,
        });

        const line = new THREE.Line(edgeGeometry, edgeMaterial);
        edgesGroup.add(line);
      }
    });

    // Create Nodes (Spheres)
    const nodesGroup = new THREE.Group();
    graphGroup.add(nodesGroup);

    const sphereGeometries: Record<number, THREE.SphereGeometry> = {};
    const nodeMeshes: { mesh: THREE.Mesh; node: NetworkNode }[] = [];

    data.nodes.forEach((node) => {
      const pos = nodePositions[node.id];
      if (!pos) return;

      // Cache geometries by size for speed
      const sizeRounded = Math.round(node.size || 6);
      if (!sphereGeometries[sizeRounded]) {
        sphereGeometries[sizeRounded] = new THREE.SphereGeometry(sizeRounded * 0.4, 16, 16);
      }
      const geom = sphereGeometries[sizeRounded];

      let colorVal = 0x2EC04A; // Low risk green
      if (node.color === '#E63946') colorVal = 0xE63946;
      else if (node.color === '#F97316') colorVal = 0xF97316;
      else if (node.color === '#EAB308') colorVal = 0xEAB308;

      const mat = new THREE.MeshPhongMaterial({
        color: colorVal,
        shininess: 80,
        specular: 0xffffff,
      });

      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(pos);
      nodesGroup.add(mesh);
      nodeMeshes.push({ mesh, node });
    });

    // Raycaster for node clicks
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const handleMouseClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(nodesGroup.children);

      if (intersects.length > 0) {
        const clickedMesh = intersects[0].object as THREE.Mesh;
        const match = nodeMeshes.find((nm) => nm.mesh === clickedMesh);
        if (match && onNodeClick) {
          onNodeClick(match.node.id);
        }
      }
    };

    renderer.domElement.addEventListener('click', handleMouseClick);

    // Track dragging/hover for rotation
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const handleMouseDown = (e: MouseEvent) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const deltaMove = {
        x: e.clientX - previousMousePosition.x,
        y: e.clientY - previousMousePosition.y,
      };

      graphGroup.rotation.y += deltaMove.x * 0.005;
      graphGroup.rotation.x += deltaMove.y * 0.005;

      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => { isDragging = false; };

    renderer.domElement.addEventListener('mousedown', handleMouseDown);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Resize Handler
    const handleResize = () => {
      if (!containerRef.current || isDestroyed.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener('resize', handleResize);

    // Animation Loop
    const animate = () => {
      if (isDestroyed.current) return;
      requestAnimationFrame(animate);

      // Slower auto rotation
      if (!isDragging) {
        graphGroup.rotation.y += 0.002;
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      isDestroyed.current = true;
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mouseup', handleMouseUp);
      if (renderer.domElement) {
        renderer.domElement.removeEventListener('click', handleMouseClick);
        renderer.domElement.removeEventListener('mousedown', handleMouseDown);
        renderer.domElement.removeEventListener('mousemove', handleMouseMove);
      }
      renderer.dispose();
    };
  }, [data, onNodeClick]);

  return (
    <div className="three-canvas-container w-full h-full min-h-[480px]" ref={containerRef}>
      <div className="absolute inset-0 flex items-center justify-center bg-black/5 flex-col pointer-events-none">
        <span className="font-display font-bold text-xs tracking-widest text-[#D4A843]/60 mb-2">BUILDING NETWORK MODEL</span>
        <span className="h-1 w-24 bg-white/10 overflow-hidden relative">
          <span className="absolute inset-y-0 bg-[#5BC0EB] w-1/2 animate-infinite-loading" />
        </span>
      </div>
    </div>
  );
};
