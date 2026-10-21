import { ReactLenis, type LenisRef } from "lenis/react";
import {
  cancelFrame,
  frame,
} from "motion/react";
import {
  type ReactNode,
  useEffect,
  useRef,
} from "react";

import "lenis/dist/lenis.css";

interface SmoothScrollProps {
  children: ReactNode;
}

export default function SmoothScroll({
  children,
}: SmoothScrollProps) {
  const lenisRef = useRef<LenisRef>(null);

  useEffect(() => {
    const update = ({
      timestamp,
    }: {
      timestamp: number;
    }) => {
      lenisRef.current?.lenis?.raf(timestamp);
    };

    frame.setup(update, true);

    return () => {
      cancelFrame(update);
    };
  }, []);

  return (
    <ReactLenis
      root
      ref={lenisRef}
      options={{
  autoRaf: false,

  // Premium but responsive desktop scrolling
  lerp: 0.085,
  duration: 1.05,

  smoothWheel: true,

  // Keep touch scrolling native/responsive
  syncTouch: false,

  // Slightly stronger wheel response
  wheelMultiplier: 0.95,
  touchMultiplier: 1.1,

}}
    >
      {children}
    </ReactLenis>
  );
}