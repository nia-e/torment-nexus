import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./style.css";

// A restarted launcher can reuse this origin. Fragment-only navigation does not
// remount React, so accept a fresh launch link by rebuilding the authenticated app.
window.addEventListener("hashchange", () => {
  if (new URLSearchParams(window.location.hash.slice(1)).get("token")) {
    window.location.reload();
  }
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
