import React, { useRef, useState, useEffect } from 'react';

export default function GlassCard3D({ title, value, prefix = "", suffix = "", subtitle, status = "neutral", tag, footer }) {
  const cardRef = useRef(null);
  const [rotateX, setRotateX] = useState(0);
  const [rotateY, setRotateY] = useState(0);
  const [glare, setGlare] = useState({ x: 50, y: 50, opacity: 0 });
  const [displayVal, setDisplayVal] = useState(0);

  useEffect(() => {
    const numTarget = typeof value === 'number' ? value : parseFloat(value) || 0;
    let start = 0;
    const duration = 1200;
    const startTime = performance.now();

    const animateCount = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayVal(start + (numTarget - start) * eased);
      if (progress < 1) requestAnimationFrame(animateCount);
    };
    requestAnimationFrame(animateCount);
  }, [value]);

  const handleMouseMove = (e) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const rotX = ((y - rect.height / 2) / (rect.height / 2)) * -9;
    const rotY = ((x - rect.width / 2) / (rect.width / 2)) * 9;
    setRotateX(rotX);
    setRotateY(rotY);
    setGlare({ x: (x / rect.width) * 100, y: (y / rect.height) * 100, opacity: 0.18 });
  };

  const handleMouseLeave = () => {
    setRotateX(0);
    setRotateY(0);
    setGlare(prev => ({ ...prev, opacity: 0 }));
  };

  const glowColors = {
    danger: "border-[#FF3D71]/40 hover:shadow-[0_0_35px_rgba(255,61,113,0.35)]",
    positive: "border-[#00F5D4]/40 hover:shadow-[0_0_35px_rgba(0,245,212,0.35)]",
    gold: "border-[#FFC857]/40 hover:shadow-[0_0_35px_rgba(255,200,87,0.35)]",
    neutral: "border-[#7B61FF]/40 hover:shadow-[0_0_35px_rgba(123,97,255,0.35)]",
  };

  return (
    <div style={{ perspective: '1200px' }} className="w-full">
      <div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          transform: `rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(0)`,
          transition: 'transform 0.12s ease-out, box-shadow 0.25s ease-out',
        }}
        className={`relative overflow-hidden rounded-2xl border bg-[#121829]/60 backdrop-blur-2xl p-6 transition-all ${glowColors[status] || glowColors.neutral}`}
      >
        <div
          className="pointer-events-none absolute inset-0 transition-opacity duration-300"
          style={{
            background: `radial-gradient(circle 280px at ${glare.x}% ${glare.y}%, rgba(123, 97, 255, ${glare.opacity}), transparent 75%)`,
          }}
        />

        <div className="relative z-10 flex flex-col justify-between h-full">
          <div className="flex items-center justify-between pb-3 border-b border-[#7B61FF]/20">
            <span className="font-mono text-xs uppercase tracking-widest text-[#E6EAF2]/60">{title}</span>
            {tag && (
              <span className="rounded-md border border-white/10 bg-white/5 px-2 py-0.5 font-mono text-[11px] text-[#00F5D4]">
                {tag}
              </span>
            )}
          </div>

          <div className="my-4">
            <div className="text-3xl font-black tracking-tight text-[#E6EAF2]">
              {prefix}
              {typeof value === 'number' && Number.isInteger(value)
                ? Math.round(displayVal)
                : displayVal.toFixed(2)}
              {suffix}
            </div>
            {subtitle && <p className="mt-1 text-xs text-[#E6EAF2]/60">{subtitle}</p>}
          </div>

          {footer && (
            <div className="border-t border-[#7B61FF]/15 pt-3 font-mono text-[11px] text-[#E6EAF2]/50">
              {footer}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
