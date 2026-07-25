'use client';

import React, { useEffect, useRef, useState } from 'react';
import { api, GeoArc } from '@/lib/api';
import * as THREE from 'three';

export const RiskGlobe: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [geoData, setGeoData] = useState<GeoArc[]>([]);
  const isDestroyed = useRef(false);

  useEffect(() => {
    api.getGeoData()
      .then((data) => {
        setGeoData(data.arcs);
      })
      .catch((err) => console.error('Failed to fetch geo data for globe', err));
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    isDestroyed.current = false;

    const width = containerRef.current.clientWidth || 500;
    const height = containerRef.current.clientHeight || 500;

    // Create scene
    const scene = new THREE.Scene();

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 250;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);

    // Globe Group
    const globeGroup = new THREE.Group();
    scene.add(globeGroup);

    // Earth Sphere
    const sphereGeo = new THREE.SphereGeometry(80, 64, 64);
    
    // Neubrutalist styling: dark blue wireframe globe with grid lines
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0x1e293b,
      wireframe: true,
      transparent: true,
      opacity: 0.15,
    });
    const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
    globeGroup.add(sphereMesh);

    // Solid core
    const coreGeo = new THREE.SphereGeometry(79.5, 32, 32);
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0x0A0A0F,
      transparent: true,
      opacity: 0.9,
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    globeGroup.add(coreMesh);

    // Points of earth (dots grid)
    const pointsGeo = new THREE.BufferGeometry();
    const positions = [];
    const radius = 80;
    
    // Create random dot points to outline the sphere
    for (let i = 0; i < 800; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);
      positions.push(x, y, z);
    }
    
    pointsGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const pointsMat = new THREE.PointsMaterial({
      color: 0xD4A843,
      size: 1.5,
      transparent: true,
      opacity: 0.4,
    });
    const pointCloud = new THREE.Points(pointsGeo, pointsMat);
    globeGroup.add(pointCloud);

    // Add lighting (even basic materials can look better with some light if tweaked)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    // Helper: Convert Lat/Lng to Vector3
    const latLngToVector3 = (lat: number, lng: number, r: number) => {
      const phi = (90 - lat) * (Math.PI / 180);
      const theta = (lng + 180) * (Math.PI / 180);

      const x = -(r * Math.sin(phi) * Math.sin(theta));
      const y = r * Math.cos(phi);
      const z = r * Math.sin(phi) * Math.cos(theta);

      return new THREE.Vector3(x, y, z);
    };

    // Draw Arcs from GeoData
    const arcObjects: THREE.Line[] = [];
    const curvePoints: THREE.Vector3[][] = [];
    
    geoData.forEach((arc) => {
      const pStart = latLngToVector3(arc.from_lat, arc.from_lng, radius);
      const pEnd = latLngToVector3(arc.to_lat, arc.to_lng, radius);

      // Interpolate middle point with height
      const midPoint = new THREE.Vector3().addVectors(pStart, pEnd).multiplyScalar(0.5);
      const dist = pStart.distanceTo(pEnd);
      
      // Arc height relative to distance
      const arcHeight = radius + dist * 0.25;
      midPoint.normalize().multiplyScalar(arcHeight);

      // Create quadratic bezier curve
      const curve = new THREE.QuadraticBezierCurve3(pStart, midPoint, pEnd);
      const points = curve.getPoints(50);
      curvePoints.push(points);

      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      
      // Color conversion
      let colorVal = 0x5BC0EB; // Info
      if (arc.color === '#E63946') colorVal = 0xE63946; // Red
      else if (arc.color === '#F97316') colorVal = 0xF97316; // Orange
      else if (arc.color === '#EAB308') colorVal = 0xEAB308; // Gold

      const lineMat = new THREE.LineBasicMaterial({
        color: colorVal,
        linewidth: 2,
        transparent: true,
        opacity: 0.8,
      });

      const line = new THREE.Line(lineGeo, lineMat);
      globeGroup.add(line);
      arcObjects.push(line);

      // Country points
      const startHotspotGeo = new THREE.SphereGeometry(1.5, 8, 8);
      const startHotspotMat = new THREE.MeshBasicMaterial({ color: colorVal });
      const startHotspot = new THREE.Mesh(startHotspotGeo, startHotspotMat);
      startHotspot.position.copy(pStart);
      globeGroup.add(startHotspot);

      const endHotspotGeo = new THREE.SphereGeometry(1.5, 8, 8);
      const endHotspotMat = new THREE.MeshBasicMaterial({ color: colorVal });
      const endHotspot = new THREE.Mesh(endHotspotGeo, endHotspotMat);
      endHotspot.position.copy(pEnd);
      globeGroup.add(endHotspot);
    });

    // Particle flow animation on arcs
    const particlesCount = curvePoints.length;
    const particleGeometry = new THREE.SphereGeometry(1.2, 8, 8);
    const particles: { mesh: THREE.Mesh; points: THREE.Vector3[]; index: number; speed: number }[] = [];

    curvePoints.forEach((points, i) => {
      const arc = geoData[i];
      let colorVal = 0x5BC0EB;
      if (arc.color === '#E63946') colorVal = 0xE63946;
      else if (arc.color === '#F97316') colorVal = 0xF97316;
      else if (arc.color === '#EAB308') colorVal = 0xEAB308;

      const particleMat = new THREE.MeshBasicMaterial({
        color: colorVal,
        transparent: true,
        opacity: 0.95,
      });
      const particleMesh = new THREE.Mesh(particleGeometry, particleMat);
      particleMesh.position.copy(points[0]);
      globeGroup.add(particleMesh);

      particles.push({
        mesh: particleMesh,
        points: points,
        index: 0,
        speed: 0.4 + Math.random() * 0.6, // random speed
      });
    });

    // Rotation variables
    let isHovered = false;
    let targetRotationX = 0;
    let targetRotationY = 0;
    let rotationX = 0;
    let rotationY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isHovered) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width * 2 - 1;
      const y = -(e.clientY - rect.top) / rect.height * 2 + 1;
      targetRotationY = x * Math.PI * 0.5;
      targetRotationX = y * Math.PI * 0.5;
    };

    const handleMouseEnter = () => { isHovered = true; };
    const handleMouseLeave = () => { isHovered = false; };

    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    renderer.domElement.addEventListener('mouseenter', handleMouseEnter);
    renderer.domElement.addEventListener('mouseleave', handleMouseLeave);

    // Window resize handler
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

      // Auto rotation when not hovered, else track mouse
      if (!isHovered) {
        globeGroup.rotation.y += 0.005;
        globeGroup.rotation.x = THREE.MathUtils.lerp(globeGroup.rotation.x, 0.1, 0.05);
      } else {
        rotationX = THREE.MathUtils.lerp(rotationX, targetRotationX, 0.05);
        rotationY = THREE.MathUtils.lerp(rotationY, targetRotationY, 0.05);
        globeGroup.rotation.x = rotationX;
        globeGroup.rotation.y = rotationY;
      }

      // Animate particles along bezier arcs
      particles.forEach((p) => {
        p.index += p.speed;
        if (p.index >= p.points.length) {
          p.index = 0;
        }
        const pointIdx = Math.floor(p.index);
        p.mesh.position.copy(p.points[pointIdx]);
      });

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      isDestroyed.current = true;
      window.removeEventListener('resize', handleResize);
      if (renderer.domElement) {
        renderer.domElement.removeEventListener('mousemove', handleMouseMove);
        renderer.domElement.removeEventListener('mouseenter', handleMouseEnter);
        renderer.domElement.removeEventListener('mouseleave', handleMouseLeave);
      }
      renderer.dispose();
    };
  }, [geoData]);

  return (
    <div className="three-canvas-container w-full h-full min-h-[400px]" ref={containerRef}>
      <div className="absolute inset-0 flex items-center justify-center bg-black/5 flex-col pointer-events-none">
        <span className="font-display font-bold text-xs tracking-widest text-[#D4A843]/60 mb-2">INITIALIZING WEBGL CORE</span>
        <span className="h-1 w-24 bg-white/10 overflow-hidden relative">
          <span className="absolute inset-y-0 bg-[#D4A843] w-1/2 animate-infinite-loading" />
        </span>
      </div>
    </div>
  );
};
