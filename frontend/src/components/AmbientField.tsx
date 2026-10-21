import {
  useEffect,
  useRef,
} from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  phase: number;
}

export default function AmbientField() {
  const canvasRef =
    useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let width = 0;
    let height = 0;
    let animationFrame = 0;

    const pointer = {
      x: -1000,
      y: -1000,
      active: false,
    };

    let particles: Particle[] = [];

    const createParticles = () => {
      const count =
        window.innerWidth < 700 ? 34 : 68;

      particles = Array.from(
        { length: count },
        (_, index) => ({
          x: Math.random() * width,
          y: Math.random() * height,
          vx:
            (Math.random() - 0.5) *
            0.12,
          vy:
            (Math.random() - 0.5) *
            0.12,
          radius:
            0.7 + Math.random() * 1.8,
          phase:
            index * 0.71 +
            Math.random() * 3,
        }),
      );
    };

    const resize = () => {
      const ratio = Math.min(
        window.devicePixelRatio || 1,
        1.5,
      );

      width = window.innerWidth;
      height = window.innerHeight;

      canvas.width =
        width * ratio;

      canvas.height =
        height * ratio;

      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      ctx.setTransform(
        ratio,
        0,
        0,
        ratio,
        0,
        0,
      );

      createParticles();
    };

    const draw = (time: number) => {
      ctx.clearRect(
        0,
        0,
        width,
        height,
      );

      const seconds = time * 0.001;

      const background =
        ctx.createRadialGradient(
          width * 0.48,
          height * 0.24,
          0,
          width * 0.48,
          height * 0.24,
          Math.max(width, height) * 0.75,
        );

      background.addColorStop(
        0,
        "rgba(34, 197, 94, 0.065)",
      );

      background.addColorStop(
        0.42,
        "rgba(16, 185, 129, 0.025)",
      );

      background.addColorStop(
        1,
        "rgba(2, 8, 5, 0)",
      );

      ctx.fillStyle = background;

      ctx.fillRect(
        0,
        0,
        width,
        height,
      );

      /*
       * Soft grid.
       */
      ctx.beginPath();

      const gridSize = 72;

      for (
        let x = 0;
        x < width;
        x += gridSize
      ) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }

      for (
        let y = 0;
        y < height;
        y += gridSize
      ) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }

      ctx.strokeStyle =
        "rgba(105, 225, 157, 0.025)";

      ctx.lineWidth = 1;

      ctx.stroke();

      /*
       * Particles.
       */
      for (const particle of particles) {
        const wave =
          Math.sin(
            seconds * 0.25 +
              particle.phase,
          ) * 0.06;

        particle.x +=
          particle.vx + wave;

        particle.y +=
          particle.vy;

        if (particle.x < -20)
          particle.x = width + 20;

        if (particle.x > width + 20)
          particle.x = -20;

        if (particle.y < -20)
          particle.y = height + 20;

        if (particle.y > height + 20)
          particle.y = -20;

        /*
         * Cursor repulsion / attraction.
         */
        if (pointer.active) {
          const dx =
            pointer.x - particle.x;

          const dy =
            pointer.y - particle.y;

          const distance =
            Math.sqrt(
              dx * dx + dy * dy,
            );

          const radius = 190;

          if (
            distance > 0 &&
            distance < radius
          ) {
            const force =
              (1 - distance / radius) *
              0.12;

            particle.vx -=
              (dx / distance) *
              force;

            particle.vy -=
              (dy / distance) *
              force;
          }
        }

        particle.vx *= 0.995;
        particle.vy *= 0.995;

        const glow =
          ctx.createRadialGradient(
            particle.x,
            particle.y,
            0,
            particle.x,
            particle.y,
            particle.radius * 8,
          );

        glow.addColorStop(
          0,
          "rgba(74, 222, 128, 0.20)",
        );

        glow.addColorStop(
          1,
          "rgba(74, 222, 128, 0)",
        );

        ctx.fillStyle = glow;

        ctx.beginPath();

        ctx.arc(
          particle.x,
          particle.y,
          particle.radius * 8,
          0,
          Math.PI * 2,
        );

        ctx.fill();

        ctx.fillStyle =
          "rgba(126, 242, 170, 0.35)";

        ctx.beginPath();

        ctx.arc(
          particle.x,
          particle.y,
          particle.radius,
          0,
          Math.PI * 2,
        );

        ctx.fill();
      }

      /*
       * Particle connections.
       */
      for (
        let i = 0;
        i < particles.length;
        i++
      ) {
        for (
          let j = i + 1;
          j < particles.length;
          j++
        ) {
          const a = particles[i];
          const b = particles[j];

          const dx = a.x - b.x;
          const dy = a.y - b.y;

          const distance =
            Math.sqrt(dx * dx + dy * dy);

          if (distance < 145) {
            const opacity =
              (1 - distance / 145) *
              0.075;

            ctx.strokeStyle =
              `rgba(88, 220, 140, ${opacity})`;

            ctx.beginPath();

            ctx.moveTo(
              a.x,
              a.y,
            );

            ctx.lineTo(
              b.x,
              b.y,
            );

            ctx.stroke();
          }
        }
      }

      if (!reducedMotion) {
        animationFrame =
          requestAnimationFrame(draw);
      }
    };

    const pointerMove = (
      event: PointerEvent,
    ) => {
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      pointer.active = true;
    };

    const pointerLeave = () => {
      pointer.active = false;
    };

    resize();

    window.addEventListener(
      "resize",
      resize,
    );

    window.addEventListener(
      "pointermove",
      pointerMove,
      { passive: true },
    );

    window.addEventListener(
      "pointerleave",
      pointerLeave,
    );

    if (reducedMotion) {
      draw(0);
    } else {
      animationFrame =
        requestAnimationFrame(draw);
    }

    return () => {
      cancelAnimationFrame(
        animationFrame,
      );

      window.removeEventListener(
        "resize",
        resize,
      );

      window.removeEventListener(
        "pointermove",
        pointerMove,
      );

      window.removeEventListener(
        "pointerleave",
        pointerLeave,
      );
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="ambient-field"
      aria-hidden="true"
    />
  );
}