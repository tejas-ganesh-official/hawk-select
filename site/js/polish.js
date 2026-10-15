// Small UI flourishes: scroll reveals, nav state on scroll, cursor spotlight on cards.
const SPOT = ".panel, .kpi, .how-card, .sample, .callout, .metric";

export function initPolish() {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // reveal on scroll
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
  document.querySelectorAll("[data-reveal]").forEach((el) => (reduce ? el.classList.add("in") : io.observe(el)));

  // denser nav once the page scrolls
  const nav = document.querySelector(".nav");
  const onScroll = () => nav?.classList.toggle("scrolled", scrollY > 24);
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // cursor spotlight: cards are created dynamically, so delegate from the document
  if (reduce || !matchMedia("(hover: hover)").matches) return;
  document.addEventListener("pointermove", (e) => {
    const card = e.target.closest?.(SPOT);
    if (!card) return;
    if (!card.classList.contains("spot")) card.classList.add("spot");
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  }, { passive: true });
}
