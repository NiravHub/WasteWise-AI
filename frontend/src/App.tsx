import "./App.css";

import {
  ArrowDown,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Leaf,
  Menu,
  Recycle,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
  Zap,
} from "lucide-react";

import {
  motion,
  useScroll,
  useSpring,
  useTransform,
} from "motion/react";

import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";

import {
  useEffect,
  useRef,
  useState,
} from "react";

import AmbientField from "./components/AmbientField";
import CursorGlow from "./components/CursorGlow";
import LiveCamera from "./components/LiveCamera";
import UploadZone from "./components/UploadZone";
import WasteAnalyzer from "./components/WasteAnalyzer";

gsap.registerPlugin(ScrollTrigger);

function App() {
  const systemRef = useRef<HTMLElement | null>(null);
  const heroRef = useRef<HTMLElement | null>(null);
  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  const [cameraOpen, setCameraOpen] =
    useState(false);

  const [menuOpen, setMenuOpen] =
    useState(false);

  /*
   * =========================================================
   * PAGE RESET
   * =========================================================
   */

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  /*
   * =========================================================
   * GSAP — SYSTEM / PIPELINE SCROLL REVEAL
   * =========================================================
   *
   * GSAP owns the pipeline scroll reveal.
   * Motion continues to control hover interactions.
   */

  useGSAP(
    () => {
      const section = systemRef.current;

      if (!section) {
        return;
      }

      const heading =
        section.querySelector(".section-heading");

      const cards =
        section.querySelectorAll(".pipeline-card");

      if (!heading || cards.length === 0) {
        return;
      }

      gsap.set(heading, {
        opacity: 0,
        y: 55,
      });

      gsap.set(cards, {
        opacity: 0,
        scale: 0.94,
      });

      /*
       * SECTION HEADING
       */

      gsap.to(heading, {
        opacity: 1,
        y: 0,
        ease: "power3.out",

        scrollTrigger: {
          trigger: section,
          start: "top 82%",
          end: "top 42%",
          scrub: 1,
          invalidateOnRefresh: true,
        },
      });

      /*
       * PIPELINE CARDS
       */

      gsap.to(cards, {
        opacity: 1,
        scale: 1,
        stagger: 0.16,
        ease: "power3.out",

        scrollTrigger: {
          trigger: section,
          start: "top 74%",
          end: "top 25%",
          scrub: 1.15,
          invalidateOnRefresh: true,
        },
      });
    },
    {
      scope: systemRef,
    },
  );

/*
 * =========================================================
 * GSAP — HERO CINEMATIC MOTION
 * =========================================================
 */

useGSAP(
  () => {
    const hero = heroRef.current;

    if (!hero) {
      return;
    }

    const content = hero.querySelector(
      ".hero-content",
    );

    const visual = hero.querySelector(
      ".hero-visual",
    );

    const eyebrow = hero.querySelector(
      ".eyebrow",
    );

    const heading = hero.querySelector(
      "h1",
    );

    const description = hero.querySelector(
      ".hero-description",
    );

    const actions = hero.querySelector(
      ".hero-actions",
    );

    const proof = hero.querySelector(
      ".hero-proof",
    );

    const scanner = hero.querySelector(
      ".scanner-frame",
    );

    const detector = hero.querySelector(
      ".card-detector",
    );

    const classifier = hero.querySelector(
      ".card-classifier",
    );

    const ood = hero.querySelector(
      ".card-ood",
    );

    const telemetry = hero.querySelector(
      ".scanner-telemetry",
    );

    if (
      !content ||
      !visual ||
      !eyebrow ||
      !heading ||
      !description ||
      !actions ||
      !proof ||
      !scanner
    ) {
      return;
    }

    /*
     * -------------------------------------------------------
     * INITIAL STATE
     * -------------------------------------------------------
     */

    gsap.set(
      [
        eyebrow,
        heading,
        description,
        actions,
        proof,
      ],
      {
        opacity: 0,
        y: 28,
      },
    );

    gsap.set(
      [
        scanner,
        detector,
        classifier,
        ood,
        telemetry,
      ].filter(Boolean),
      {
        opacity: 0,
        scale: 0.94,
      },
    );

    /*
     * -------------------------------------------------------
     * HERO ENTRANCE
     * -------------------------------------------------------
     */

    const entrance = gsap.timeline({
      defaults: {
        ease: "power3.out",
      },
    });

    entrance
      .to(
        eyebrow,
        {
          opacity: 1,
          y: 0,
          duration: 0.65,
        },
        0.05,
      )
      .to(
        heading,
        {
          opacity: 1,
          y: 0,
          duration: 0.85,
        },
        0.14,
      )
      .to(
        description,
        {
          opacity: 1,
          y: 0,
          duration: 0.7,
        },
        0.28,
      )
      .to(
        actions,
        {
          opacity: 1,
          y: 0,
          duration: 0.65,
        },
        0.38,
      )
      .to(
        proof,
        {
          opacity: 1,
          y: 0,
          duration: 0.55,
        },
        0.48,
      )
      .to(
        scanner,
        {
          opacity: 1,
          scale: 1,
          duration: 1.05,
          ease: "power3.out",
        },
        0.18,
      )
      .to(
        [detector, classifier, ood].filter(Boolean),
        {
          opacity: 1,
          scale: 1,
          duration: 0.7,
          stagger: 0.1,
        },
        0.62,
      )
      .to(
        telemetry ? [telemetry] : [],
        {
          opacity: 1,
          scale: 1,
          duration: 0.55,
        },
        0.78,
      );

    /*
     * -------------------------------------------------------
     * SCROLL CHOREOGRAPHY
     * -------------------------------------------------------
     */

    const scrollTimeline = gsap.timeline({
      scrollTrigger: {
        trigger: hero,
        start: "top top",
        end: "bottom top",
        scrub: 1.2,
        invalidateOnRefresh: true,
      },
    });

    scrollTimeline
      .to(
        content,
        {
          y: -70,
          opacity: 0.45,
          ease: "none",
        },
        0,
      )
      .to(
        visual,
        {
          y: -45,
          scale: 0.92,
          opacity: 0.6,
          ease: "none",
        },
        0,
      )
      .to(
        scanner,
        {
          rotateX: 2,
          ease: "none",
        },
        0,
      );

    /*
     * -------------------------------------------------------
     * FLOATING CARD PARALLAX
     * -------------------------------------------------------
     */

    if (detector) {
      gsap.to(detector, {
        x: -18,
        y: -24,
        scrollTrigger: {
          trigger: hero,
          start: "top top",
          end: "bottom top",
          scrub: 1.4,
        },
      });
    }

    if (classifier) {
      gsap.to(classifier, {
        x: 20,
        y: 18,
        scrollTrigger: {
          trigger: hero,
          start: "top top",
          end: "bottom top",
          scrub: 1.6,
        },
      });
    }

    if (ood) {
      gsap.to(ood, {
        x: 24,
        y: -18,
        scrollTrigger: {
          trigger: hero,
          start: "top top",
          end: "bottom top",
          scrub: 1.5,
        },
      });
    }
  },
  {
    scope: heroRef,
  },
);



  /*
   * =========================================================
   * PAGE SCROLL PROGRESS
   * =========================================================
   */

  const { scrollYProgress } = useScroll();

  const smoothProgress = useSpring(
    scrollYProgress,
    {
      stiffness: 110,
      damping: 30,
      restDelta: 0.001,
    },
  );

  const heroY = useTransform(
    scrollYProgress,
    [0, 0.25],
    [0, -80],
  );

  const heroOpacity = useTransform(
    scrollYProgress,
    [0, 0.22],
    [1, 0.45],
  );

  /*
   * =========================================================
   * PREVIEW URL CLEANUP
   * =========================================================
   */

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  /*
   * =========================================================
   * FILE HANDLING
   * =========================================================
   */

  const handleFileSelected = (
    file: File,
  ) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);

    setPreviewUrl(
      URL.createObjectURL(file),
    );

    setTimeout(() => {
      document
        .getElementById("analysis")
        ?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
    }, 80);
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setPreviewUrl(null);
  };

  /*
   * =========================================================
   * NAVIGATION
   * =========================================================
   */

  const scrollTo = (id: string) => {
    document
      .getElementById(id)
      ?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });

    setMenuOpen(false);
  };

  /*
   * =========================================================
   * RENDER
   * =========================================================
   */

  return (
    <div
      className="app"
      id="top"
    >
      <AmbientField />

      <CursorGlow />

      <motion.div
        className="scroll-progress"
        style={{
          scaleX: smoothProgress,
        }}
      />

      {/* =====================================================
          HEADER
          ===================================================== */}

      <header className="site-header">
        <div className="container header-inner">

          <button
            className="brand"
            type="button"
            onClick={() =>
              scrollTo("top")
            }
            data-cursor="interactive"
          >
            <span className="brand-mark">
              <Leaf size={19} />
            </span>

            <span className="brand-copy">
              <strong>
                WasteWise
              </strong>

              <small>
                AI INTELLIGENCE
              </small>
            </span>
          </button>

          <nav className="desktop-nav">

            <button
              onClick={() =>
                scrollTo("top")
              }
              data-cursor="interactive"
            >
              Home
            </button>

            <button
              onClick={() =>
                scrollTo("how-it-works")
              }
              data-cursor="interactive"
            >
              System
            </button>

            <button
              onClick={() =>
                scrollTo("analysis")
              }
              data-cursor="interactive"
            >
              Analyzer
            </button>

          </nav>

          <div className="header-actions">

            <div className="live-pill">
              <span />
              AI ONLINE
            </div>

            <button
              className="header-analyze"
              data-magnetic
              onClick={() =>
                scrollTo("analysis")
              }
              data-cursor="interactive"
            >
              Analyze

              <ArrowUpRight
                size={15}
              />
            </button>

            <button
              className="mobile-menu"
              type="button"
              onClick={() =>
                setMenuOpen(
                  (value) => !value,
                )
              }
              aria-label="Open menu"
              data-cursor="interactive"
            >
              {menuOpen ? (
                <X size={20} />
              ) : (
                <Menu size={20} />
              )}
            </button>

          </div>
        </div>

        {menuOpen && (
          <motion.div
            className="mobile-nav"
            initial={{
              opacity: 0,
              y: -12,
            }}
            animate={{
              opacity: 1,
              y: 0,
            }}
          >
            <button
              onClick={() =>
                scrollTo("top")
              }
            >
              Home
            </button>

            <button
              onClick={() =>
                scrollTo("how-it-works")
              }
            >
              System
            </button>

            <button
              onClick={() =>
                scrollTo("analysis")
              }
            >
              Analyzer
            </button>
          </motion.div>
        )}
      </header>

      <main>

        {/* =====================================================
            HERO
            ===================================================== */}

        <section
ref={heroRef}
          id="hero"
          className="hero"
        >
          <motion.div
            className="container hero-grid"
            style={{
              y: heroY,
              opacity: heroOpacity,
            }}
          >

            {/* =================================================
                HERO CONTENT
                ================================================= */}

            <div className="hero-content">

              {/* =================================================
                  HERO EYEBROW
                  ================================================= */}

              <div className="eyebrow">
                <span className="eyebrow-dot" />

                <span className="eyebrow-status">
                  SYSTEM ONLINE
                </span>

                <span className="eyebrow-divider" />

                Intelligent waste perception

                <ChevronRight size={13} />
              </div>

              {/* =================================================
                  HERO TITLE
                  ================================================= */}

              <h1>
                See waste.
                <br />

                <span>
                  Understand it.
                </span>

                <br />

                Act{" "}
                <em>
                  smarter.
                </em>
              </h1>

              {/* =================================================
                  HERO DESCRIPTION
                  ================================================= */}

              <p className="hero-description">
                WasteWise AI combines
                object detection,
                neural-network
                classification, OOD
                verification, and semantic
                filtering into one
                intelligent waste-analysis
                pipeline.
              </p>

              {/* =================================================
                  HERO ACTIONS
                  ================================================= */}

              <div className="hero-actions">

                <button
                  className="primary-button"
                  data-magnetic
                  onClick={() =>
                    scrollTo("analysis")
                  }
                  data-cursor="interactive"
                >
                  <Sparkles size={18} />

                  Start analyzing

                  <ArrowUpRight size={16} />
                </button>

                <button
                  className="ghost-button"
                  data-magnetic
                  onClick={() =>
                    scrollTo("how-it-works")
                  }
                  data-cursor="interactive"
                >
                  Explore the system

                  <ArrowDown size={15} />
                </button>

              </div>

              {/* =================================================
                  HERO PROOF
                  ================================================= */}

              <div className="hero-proof">

                <div className="hero-proof-item">
                  <span className="hero-proof-index">
                    01
                  </span>

                  <CheckCircle2
                    size={15}
                  />

                  <span>
                    19-class intelligence
                  </span>
                </div>

                <div className="hero-proof-item">
                  <span className="hero-proof-index">
                    02
                  </span>

                  <ShieldCheck
                    size={15}
                  />

                  <span>
                    OOD verified
                  </span>
                </div>

                <div className="hero-proof-item">
                  <span className="hero-proof-index">
                    03
                  </span>

                  <Zap
                    size={15}
                  />

                  <span>
                    GPU inference
                  </span>
                </div>

              </div>

            </div>

            {/* =================================================
                HERO SCANNER
                ================================================= */}

            <motion.div
              className="hero-visual"
              initial={{
                opacity: 0,
                scale: 0.92,
              }}
              animate={{
                opacity: 1,
                scale: 1,
              }}
              transition={{
                duration: 1.1,
                delay: 0.18,
              }}
            >

              {/* Ambient depth */}

              <div className="hero-glow glow-one" />

              <div className="hero-glow glow-two" />

              {/* Orbital intelligence rings */}

              <motion.div
                className="orbit orbit-one"
                animate={{
                  rotate: 360,
                }}
                transition={{
                  duration: 32,
                  repeat: Infinity,
                  ease: "linear",
                }}
              />

              <motion.div
                className="orbit orbit-two"
                animate={{
                  rotate: -360,
                }}
                transition={{
                  duration: 42,
                  repeat: Infinity,
                  ease: "linear",
                }}
              />

              {/* Scanner */}

              <div className="scanner-frame">

                <span className="scanner-corner scanner-corner--tl" />
                <span className="scanner-corner scanner-corner--tr" />
                <span className="scanner-corner scanner-corner--bl" />
                <span className="scanner-corner scanner-corner--br" />

                {/* Scanner label */}

                <div className="scanner-label scanner-label--top">

                  <span className="scanner-live-dot" />

                  <span>
                    LIVE
                  </span>

                  <strong>
                    WASTE SCANNER
                  </strong>

                  <small>
                    / 01
                  </small>

                </div>

                {/* Target */}

                <div className="scanner-target">

                  <span className="target-cross target-cross--top" />
                  <span className="target-cross target-cross--right" />
                  <span className="target-cross target-cross--bottom" />
                  <span className="target-cross target-cross--left" />

                </div>

                {/* AI Core */}

                <motion.div
                  className="ai-core"
                  animate={{
                    y: [0, -9, 0],
                    rotateX: [
                      0,
                      2,
                      0,
                    ],
                    rotateY: [
                      0,
                      -4,
                      0,
                    ],
                  }}
                  transition={{
                    duration: 6,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                >

                  <div className="core-scan" />

                  <div className="core-icon">
                    <BrainCircuit
                      size={38}
                      strokeWidth={1.35}
                    />
                  </div>

                  <span className="core-label">
                    WASTEWISE
                  </span>

                  <strong>
                    PERCEPTION
                  </strong>

                  <div className="core-status">
                    <span />
                    SCANNER READY
                  </div>

                  <div className="core-meta">
                    <span>
                      YOLO11s
                    </span>

                    <b>•</b>

                    <span>
                      RESNET50
                    </span>

                    <b>•</b>

                    <span>
                      OOD
                    </span>
                  </div>

                </motion.div>

                {/* Left readout */}

                <div className="scanner-readout scanner-readout--left">

                  <span>
                    MODE
                  </span>

                  <strong>
                    OBJECT DETECTION
                  </strong>

                </div>

                {/* Right readout */}

                <div className="scanner-readout scanner-readout--right">

                  <span>
                    VERIFY
                  </span>

                  <strong>
                    OOD + SEMANTIC
                  </strong>

                </div>

                {/* Technical telemetry */}

                <div className="scanner-telemetry">

                  <div>
                    <span>
                      DETECT
                    </span>

                    <strong>
                      YOLO11s
                    </strong>
                  </div>

                  <div>
                    <span>
                      CLASSIFY
                    </span>

                    <strong>
                      RESNET50
                    </strong>
                  </div>

                  <div>
                    <span>
                      VERIFY
                    </span>

                    <strong>
                      OOD + GATE
                    </strong>
                  </div>

                </div>

                <div className="scanner-scanline" />

              </div>

              {/* =================================================
                  FLOATING DETECTOR CARD
                  ================================================= */}

              <motion.div
                className="floating-card card-detector"
                whileHover={{
                  y: -5,
                  scale: 1.025,
                }}
                transition={{
                  duration: 0.25,
                }}
              >
                <span>
                  DETECTOR
                </span>

                <strong>
                  YOLO11s
                </strong>

                <small>
                  6 practical classes
                </small>
              </motion.div>

              {/* =================================================
                  FLOATING CLASSIFIER CARD
                  ================================================= */}

              <motion.div
                className="floating-card card-classifier"
                whileHover={{
                  y: -5,
                  scale: 1.025,
                }}
                transition={{
                  duration: 0.25,
                }}
              >
                <span>
                  CLASSIFIER
                </span>

                <strong>
                  ResNet50
                </strong>

                <small>
                  19 detailed classes
                </small>
              </motion.div>

              {/* =================================================
                  FLOATING OOD CARD
                  ================================================= */}

              <motion.div
                className="floating-card card-ood"
                whileHover={{
                  y: -5,
                  scale: 1.025,
                }}
                transition={{
                  duration: 0.25,
                }}
              >

                <ShieldCheck
                  size={16}
                />

                <div>

                  <strong>
                    OOD
                  </strong>

                  <small>
                    Verified
                  </small>

                </div>

              </motion.div>

            </motion.div>

          </motion.div>

          {/* =====================================================
              HERO BOTTOM
              ===================================================== */}

          <div className="hero-bottom">

            <div className="container hero-bottom-inner">

              <span>
                SCROLL TO EXPLORE
              </span>

              <motion.div
                animate={{
                  y: [0, 7, 0],
                }}
                transition={{
                  duration: 1.7,
                  repeat: Infinity,
                }}
              >
                <ArrowDown
                  size={17}
                />
              </motion.div>

              <span className="hero-line" />

            </div>

          </div>

        </section>

        {/* =====================================================
            SYSTEM / PIPELINE
            ===================================================== */}

        <section
          ref={systemRef}
          id="how-it-works"
          className="system-section"
        >
          <div className="container">

            <div className="section-heading">

              <span>
                THE PIPELINE
              </span>

              <h2>
                From pixels
                <br />
                to decisions.
              </h2>

              <p>
                Every prediction passes
                through multiple intelligence
                layers before WasteWise
                presents a final result.
              </p>

            </div>

            <div className="pipeline">

              {[
                {
                  number: "01",
                  icon: Upload,
                  title: "Capture",
                  text: "Upload an image or use the live camera.",
                },
                {
                  number: "02",
                  icon: ScanSearch,
                  title: "Detect",
                  text: "YOLO11s finds practical waste candidates.",
                },
                {
                  number: "03",
                  icon: BrainCircuit,
                  title: "Classify",
                  text: "ResNet50 interprets each detected object.",
                },
                {
                  number: "04",
                  icon: ShieldCheck,
                  title: "Verify",
                  text: "OOD and semantic compatibility checks reject unreliable results.",
                },
              ].map((step) => {
                const Icon = step.icon;

                return (
                  <motion.div
                    key={step.number}
                    className="pipeline-card"
                    whileHover={{
                      y: -8,
                      transition: {
                        duration: 0.25,
                      },
                    }}
                  >

                    <div className="pipeline-top">

                      <span>
                        {step.number}
                      </span>

                      <Icon
                        size={21}
                      />

                    </div>

                    <h3>
                      {step.title}
                    </h3>

                    <p>
                      {step.text}
                    </p>

                    <div className="pipeline-bar">
                      <span />
                    </div>

                  </motion.div>
                );
              })}

            </div>

            <div className="architecture-strip">

              <div>
                <Cpu size={18} />

                <span>
                  YOLO11s
                </span>
              </div>

              <ChevronRight />

              <div>
                <BrainCircuit
                  size={18}
                />

                <span>
                  ResNet50 Stage 2
                </span>
              </div>

              <ChevronRight />

              <div>
                <ShieldCheck
                  size={18}
                />

                <span>
                  OOD
                </span>
              </div>

              <ChevronRight />

              <div>
                <Recycle size={18} />

                <span>
                  Decision
                </span>
              </div>

            </div>

          </div>
        </section>

        {/* =====================================================
            ANALYZER
            ===================================================== */}

        <section
          id="analysis"
          className="analyzer-section"
        >
          <div className="container">

            <motion.div
              className="analyzer-heading"
              initial={{
                opacity: 0,
                y: 30,
              }}
              whileInView={{
                opacity: 1,
                y: 0,
              }}
              viewport={{
                once: true,
              }}
            >

              <div>

                <span>
                  LIVE INTELLIGENCE
                </span>

                <h2>
                  Waste analyzer
                </h2>

              </div>

              <div className="analyzer-live">
                <span />
                Pipeline operational
              </div>

            </motion.div>

            <motion.div
              className="upload-shell"
              initial={{
                opacity: 0,
                y: 30,
              }}
              whileInView={{
                opacity: 1,
                y: 0,
              }}
              viewport={{
                once: true,
              }}
            >

              <UploadZone
                onFileSelected={
                  handleFileSelected
                }
                onCameraClick={() =>
                  setCameraOpen(true)
                }
              />

            </motion.div>

            <div className="analysis-anchor">

              <WasteAnalyzer
                selectedFile={
                  selectedFile
                }
                previewUrl={
                  previewUrl
                }
                onClear={
                  handleClear
                }
              />

            </div>

          </div>
        </section>

        {/* =====================================================
            CLOSING
            ===================================================== */}

        <section className="closing-section">

          <div className="container closing-card">

            <div className="closing-glow" />

            <div className="closing-content">

              <div className="closing-icon">
                <Recycle size={24} />
              </div>

              <span>
                WASTEWISE AI
              </span>

              <h2>
                Better recognition.
                <br />
                Better decisions.
              </h2>

              <p>
                Intelligent perception designed
                to make waste classification
                clearer, safer, and more useful.
              </p>

              <button
                className="primary-button"
                onClick={() =>
                  scrollTo("analysis")
                }
                data-cursor="interactive"
              >
                Analyze another image

                <ArrowUpRight
                  size={16}
                />
              </button>

            </div>

          </div>

        </section>

      </main>

      {/* =====================================================
          FOOTER
          ===================================================== */}

      <footer className="site-footer">

        <div className="container footer-inner">

          <div className="footer-brand">

            <div className="brand-mark small">
              <Leaf size={15} />
            </div>

            <div>

              <strong>
                WasteWise AI
              </strong>

              <span>
                Intelligent waste perception
              </span>

            </div>

          </div>

          <div className="footer-meta">

            YOLO11s

            <span>•</span>

            ResNet50

            <span>•</span>

            OOD

            <span>•</span>

            Semantic verification

          </div>

          <button
            className="back-top"
            onClick={() =>
              scrollTo("top")
            }
            data-cursor="interactive"
          >
            Back to top

            <ArrowUpRight
              size={14}
            />
          </button>

        </div>

      </footer>

      {/* =====================================================
          LIVE CAMERA
          ===================================================== */}

      {cameraOpen && (
        <LiveCamera
          onClose={() =>
            setCameraOpen(false)
          }
        />
      )}

    </div>
  );
}

export default App;