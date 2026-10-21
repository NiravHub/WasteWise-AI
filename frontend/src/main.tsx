import {
  StrictMode,
} from "react";

import {
  createRoot,
} from "react-dom/client";

import App from "./App";
import SmoothScroll from "./components/SmoothScroll";

import "./index.css";

createRoot(
  document.getElementById(
    "root",
  )!,
).render(
  <StrictMode>
    <SmoothScroll>
      <App />
    </SmoothScroll>
  </StrictMode>,
);