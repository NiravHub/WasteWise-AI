import {
  useEffect,
  useRef,
} from "react";

export default function CursorGlow() {
  const dotRef =
    useRef<HTMLDivElement>(null);

  const ringRef =
    useRef<HTMLDivElement>(null);

  const labelRef =
    useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (
      !window.matchMedia(
        "(pointer: fine)",
      ).matches
    ) {
      return;
    }

    const dot = dotRef.current;
    const ring = ringRef.current;
    const label = labelRef.current;

    const magneticElements =
      Array.from(
        document.querySelectorAll<HTMLElement>(
          "[data-magnetic]",
        ),
      );

    const magneticState =
      new Map<
        HTMLElement,
        {
          x: number;
          y: number;
        }
      >();

    magneticElements.forEach(
      (element) => {
        magneticState.set(
          element,
          {
            x: 0,
            y: 0,
          },
        );
      },
    );

    const magneticMove = (
      event: PointerEvent,
    ) => {
      magneticElements.forEach(
        (element) => {
          const rect =
            element.getBoundingClientRect();

          const centerX =
            rect.left +
            rect.width / 2;

          const centerY =
            rect.top +
            rect.height / 2;

          const distanceX =
            event.clientX - centerX;

          const distanceY =
            event.clientY - centerY;

          const distance =
            Math.sqrt(
              distanceX * distanceX +
              distanceY * distanceY,
            );

          const radius =
            Math.max(
              rect.width,
              rect.height,
            ) * 1.15;

          const state =
            magneticState.get(
              element,
            );

          if (!state) return;

          if (distance < radius) {
            const strength =
              1 -
              distance / radius;

            state.x =
              distanceX *
              strength *
              0.16;

            state.y =
              distanceY *
              strength *
              0.16;
          } else {
            state.x *= 0.72;
            state.y *= 0.72;
          }

          element.style.transform =
            `translate3d(${state.x}px, ${state.y}px, 0)`;
        },
      );
    };

    if (!dot || !ring || !label) {
      return;
    }

    let targetX = -100;
    let targetY = -100;

    let currentX = targetX;
    let currentY = targetY;

    let labelX = targetX;
    let labelY = targetY;

    let animationFrame = 0;

    const move = (
      event: PointerEvent,
    ) => {
      targetX = event.clientX;
      targetY = event.clientY;

      dot.style.transform =
        `translate3d(${targetX}px, ${targetY}px, 0) translate(-50%, -50%)`;
    };

    const hover = (
      event: PointerEvent,
    ) => {
      const element =
        event.target as HTMLElement;

      const interactive =
        element.closest(
          "button, a, input, [data-cursor='interactive']",
        );

      const scanner =
        element.closest(
          ".scanner-frame, .hero-visual, .ww-image-stage",
        );

      ring.classList.toggle(
        "is-active",
        Boolean(interactive),
      );

      ring.classList.toggle(
        "is-scanning",
        Boolean(scanner),
      );

      label.classList.toggle(
        "is-visible",
        Boolean(scanner),
      );

      if (scanner) {
        label.textContent =
          "SCAN";
      } else if (interactive) {
        label.textContent =
          "OPEN";
      } else {
        label.textContent = "";
      }
    };

    const animate = () => {
      currentX +=
        (targetX - currentX) * 0.16;

      currentY +=
        (targetY - currentY) * 0.16;

      labelX +=
        (targetX - labelX) * 0.12;

      labelY +=
        (targetY - labelY) * 0.12;

      ring.style.transform =
        `translate3d(${currentX}px, ${currentY}px, 0) translate(-50%, -50%)`;

      label.style.transform =
        `translate3d(${labelX + 18}px, ${labelY + 18}px, 0)`;

      animationFrame =
        requestAnimationFrame(
          animate,
        );
    };

    window.addEventListener(
  "pointermove",
  move,
  { passive: true },
);

window.addEventListener(
  "pointermove",
  magneticMove,
  { passive: true },
);

    window.addEventListener(
      "pointerover",
      hover,
      { passive: true },
    );

    animationFrame =
      requestAnimationFrame(
        animate,
      );

    return () => {
      window.removeEventListener(
        "pointermove",
        magneticMove,
      );

      magneticElements.forEach(
        (element) => {
          element.style.transform =
            "";
        },
      );
      cancelAnimationFrame(
        animationFrame,
      );

      window.removeEventListener(
        "pointermove",
        move,
      );

      window.removeEventListener(
        "pointerover",
        hover,
      );
    };
  }, []);

  return (
    <>
      <div
        ref={ringRef}
        className="cursor-ring"
        aria-hidden="true"
      >
        <span className="cursor-cross cursor-cross--top" />
        <span className="cursor-cross cursor-cross--right" />
        <span className="cursor-cross cursor-cross--bottom" />
        <span className="cursor-cross cursor-cross--left" />
      </div>

      <div
        ref={dotRef}
        className="cursor-dot"
        aria-hidden="true"
      />

      <div
        ref={labelRef}
        className="cursor-label"
        aria-hidden="true"
      />
    </>
  );
}