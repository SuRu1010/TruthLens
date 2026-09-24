'use client';

import React, { useEffect, useRef } from 'react';

interface LiquidLinesProps {
  lineColor?: string;
  lineCount?: number;
  speed?: number;
  amplitudeMultiplier?: number;
  className?: string;
}

export const LiquidLinesCanvas: React.FC<LiquidLinesProps> = ({
  lineColor = 'rgba(255, 255, 255, 0.2)',
  lineCount = 7,
  speed = 0.008,
  amplitudeMultiplier = 1.0,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let step = 0;

    // Auto-adjust canvas viewport to match parent node dimensions
    const resizeCanvas = () => {
      if (!canvas.parentElement) return;
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Render loop calculating organic wave curves
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      step += speed;

      for (let i = 0; i < lineCount; i++) {
        ctx.beginPath();

        // Dynamic opacity fade per line index for depth
        const opacity = Math.max(0.05, 0.35 - i * 0.04);
        ctx.strokeStyle = lineColor.startsWith('rgba')
          ? lineColor
          : `rgba(255, 255, 255, ${opacity})`;
        ctx.lineWidth = 1.5;

        const verticalOffset = (canvas.height / (lineCount + 1)) * (i + 1);
        const freq = 0.002 + i * 0.0004;
        const amp = (25 + i * 12) * amplitudeMultiplier;

        ctx.moveTo(0, verticalOffset);

        for (let x = 0; x <= canvas.width; x += 4) {
          // Combined harmonic sines and cosines to create fluid dynamic motion
          const wave1 = Math.sin(x * freq + step + i * 0.5) * amp;
          const wave2 = Math.cos(x * 0.0015 - step * 0.7) * (amp * 0.4);
          const y = verticalOffset + wave1 + wave2;

          ctx.lineTo(x, y);
        }

        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, [lineColor, lineCount, speed, amplitudeMultiplier]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full pointer-events-none z-0 ${className}`}
    />
  );
};
