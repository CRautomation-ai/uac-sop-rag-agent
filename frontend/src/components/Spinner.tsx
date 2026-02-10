import React from "react";

export const Spinner: React.FC<{ className?: string }> = ({ className }) => (
  <span className={`spinner ${className ?? ""}`} aria-hidden />
);

export default Spinner;
