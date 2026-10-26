import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function FinTech3DCore({ distressScore = 0.0033, tier = "Baseline Propensity Tier" }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.1, 1000);
    camera.position.z = 7;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const coreGroup = new THREE.Group();
    scene.add(coreGroup);

    const isElevated = tier && tier.toLowerCase().includes("elevated");
    const primaryColor = isElevated ? 0xff4d6d : 0x00f5d4;
    const secondaryColor = 0x38bdf8;
    const goldColor = 0xf59e0b;

    const sphereGeo = new THREE.IcosahedronGeometry(2, 2);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: primaryColor,
      wireframe: true,
      transparent: true,
      opacity: 0.25,
    });
    const riskSphere = new THREE.Mesh(sphereGeo, sphereMat);
    coreGroup.add(riskSphere);

    const pointsMat = new THREE.PointsMaterial({
      color: primaryColor,
      size: 0.08,
      transparent: true,
      opacity: 0.85,
    });
    const riskNodes = new THREE.Points(sphereGeo, pointsMat);
    coreGroup.add(riskNodes);

    const ringGeo1 = new THREE.TorusGeometry(2.6, 0.02, 16, 100);
    const ringMat1 = new THREE.MeshBasicMaterial({ color: secondaryColor, transparent: true, opacity: 0.55 });
    const orbitRing1 = new THREE.Mesh(ringGeo1, ringMat1);
    orbitRing1.rotation.x = Math.PI / 3;
    coreGroup.add(orbitRing1);

    const ringGeo2 = new THREE.TorusGeometry(3.1, 0.015, 16, 100);
    const ringMat2 = new THREE.MeshBasicMaterial({ color: goldColor, transparent: true, opacity: 0.4 });
    const orbitRing2 = new THREE.Mesh(ringGeo2, ringMat2);
    orbitRing2.rotation.y = Math.PI / 4;
    orbitRing2.rotation.x = -Math.PI / 6;
    coreGroup.add(orbitRing2);

    const curvePoints = [];
    for (let i = 0; i <= 30; i++) {
      const theta = (i / 30) * Math.PI * 2;
      const r = 2.4 + Math.sin(i * 0.8) * 0.35;
      curvePoints.push(new THREE.Vector3(
        Math.cos(theta) * r,
        Math.sin(i * 0.5) * 0.8,
        Math.sin(theta) * r
      ));
    }
    const spline = new THREE.CatmullRomCurve3(curvePoints);
    const tubeGeo = new THREE.TubeGeometry(spline, 64, 0.03, 8, true);
    const tubeMat = new THREE.MeshBasicMaterial({
      color: isElevated ? 0xff4d6d : 0x06b6d4,
      transparent: true,
      opacity: 0.75,
    });
    const forecastRibbon = new THREE.Mesh(tubeGeo, tubeMat);
    coreGroup.add(forecastRibbon);

    const particleCount = 200;
    const particleGeo = new THREE.BufferGeometry();
    const particleCoords = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i++) {
      particleCoords[i] = (Math.random() - 0.5) * 9;
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(particleCoords, 3));
    const particleMat = new THREE.PointsMaterial({
      color: goldColor,
      size: 0.035,
      transparent: true,
      opacity: 0.5,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    let mouseX = 0;
    let mouseY = 0;
    const handleMouseMove = (event) => {
      const rect = mount.getBoundingClientRect();
      const x = event.clientX - rect.left - rect.width / 2;
      const y = event.clientY - rect.top - rect.height / 2;
      mouseX = (x / rect.width) * 2;
      mouseY = (y / rect.height) * 2;
    };
    mount.addEventListener('mousemove', handleMouseMove);

    const handleResize = () => {
      if (!mount) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    let animationFrameId;
    let clock = new THREE.Clock();
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      riskSphere.rotation.y = elapsed * 0.2;
      riskSphere.rotation.x = elapsed * 0.1;
      riskNodes.rotation.y = elapsed * 0.2;
      riskNodes.rotation.x = elapsed * 0.1;

      orbitRing1.rotation.z = elapsed * 0.25;
      orbitRing2.rotation.z = -elapsed * 0.18;
      forecastRibbon.rotation.y = -elapsed * 0.15;
      particles.rotation.y = elapsed * 0.04;

      const targetRotationY = mouseX * 0.6;
      const targetRotationX = -mouseY * 0.6;
      coreGroup.rotation.y += (targetRotationY - coreGroup.rotation.y) * 0.06;
      coreGroup.rotation.x += (targetRotationX - coreGroup.rotation.x) * 0.06;

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      if (mount) {
        mount.removeEventListener('mousemove', handleMouseMove);
        if (renderer.domElement.parentNode === mount) {
          mount.removeChild(renderer.domElement);
        }
      }
      renderer.dispose();
    };
  }, [tier, distressScore]);

  return (
    <div className="relative w-full h-[360px] flex items-center justify-center overflow-hidden rounded-2xl border border-cyan-500/30 bg-slate-950/80 backdrop-blur-2xl shadow-[0_0_50px_rgba(6,182,212,0.15)]">
      <div ref={mountRef} className="absolute inset-0 cursor-grab active:cursor-grabbing" />

      <div className="pointer-events-none absolute top-4 left-4 z-10 flex flex-col gap-1 font-mono text-[11px] text-cyan-400">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400 animate-ping" />
          <span className="font-semibold tracking-wider text-slate-100">FP&A 3D RISK TOPOLOGY</span>
        </div>
        <span className="text-slate-400">TENSOR: 5-DIMENSIONAL LATTICE</span>
      </div>

      <div className="pointer-events-none absolute top-4 right-4 z-10 flex flex-col items-end font-mono text-[11px]">
        <span className="text-slate-400 tracking-wider">PROPENSITY SCORE</span>
        <span className="text-lg font-bold text-cyan-300 tracking-wider">
          {(distressScore * 100).toFixed(2)}%
        </span>
      </div>

      <div className="pointer-events-none absolute bottom-4 left-4 z-10 font-mono text-[10px] text-slate-500 tracking-widest">
        ROTATION: ORBITAL 3D // TREE_SHAP COUPLING ACTIVE
      </div>

      <div className="pointer-events-none absolute bottom-4 right-4 z-10 font-mono text-[10px] text-cyan-400 border border-cyan-500/30 px-2 py-0.5 rounded bg-cyan-950/60 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
        LIVE INFERENCE
      </div>
    </div>
  );
}
