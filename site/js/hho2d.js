// Continuous HHO (Heidari et al. 2019) on a 2D landscape, stepped one iteration at a time
// so the hero canvas (and later the algorithm explainers) can animate it.

export const STRATEGY = {
  PERCH_RANDOM: 0,   // |E| >= 1, q >= 0.5: perch relative to a random hawk
  PERCH_MEAN: 1,     // |E| >= 1, q <  0.5: perch relative to rabbit and population mean
  SOFT: 2,           // r >= 0.5, |E| >= 0.5
  HARD: 3,           // r >= 0.5, |E| <  0.5
  SOFT_DIVE: 4,      // r <  0.5, |E| >= 0.5 (Levy rapid dives)
  HARD_DIVE: 5,      // r <  0.5, |E| <  0.5 (Levy rapid dives)
};

export const STRATEGY_NAMES = [
  "Perch: random hawk", "Perch: family mean", "Soft besiege",
  "Hard besiege", "Soft + rapid dives", "Hard + rapid dives",
];

// Shifted Rastrigin: many local minima, global min 0 at SHIFT (off-centre to avoid centre bias).
const SHIFT = [0.9, 2.6];
export function landscape(x, y) {
  const a = x - SHIFT[0], b = y - SHIFT[1];
  return 20 + a * a + b * b - 10 * (Math.cos(2 * Math.PI * a) + Math.cos(2 * Math.PI * b));
}
export const BOUNDS = [-5.12, 5.12];
export const OPTIMUM = SHIFT;

// Mantegna's algorithm, beta = 1.5 -> sigma ~= 0.6966
const BETA = 1.5, SIGMA = 0.6965745;
function gauss() {
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const levy = () => (0.01 * gauss() * SIGMA) / Math.pow(Math.abs(gauss()), 1 / BETA);

const clamp = (v) => Math.min(BOUNDS[1], Math.max(BOUNDS[0], v));

export class HHO2D {
  constructor({ hawks = 14, iterations = 120, f = landscape } = {}) {
    this.n = hawks;
    this.T = iterations;
    this.f = f;
    this.reset();
  }

  reset() {
    const [lb, ub] = BOUNDS;
    this.t = 0;
    this.X = Array.from({ length: this.n }, () => [lb + Math.random() * (ub - lb), lb + Math.random() * (ub - lb)]);
    this.strategy = new Array(this.n).fill(STRATEGY.PERCH_RANDOM);
    this.E = 2;
    this.Emax = 2;
    this.rabbit = null;
    this.rabbitFit = Infinity;
    this.history = [];
    this._updateRabbit();
  }

  get done() { return this.t >= this.T; }

  _updateRabbit() {
    for (const x of this.X) {
      const fx = this.f(x[0], x[1]);
      if (fx < this.rabbitFit) { this.rabbitFit = fx; this.rabbit = [x[0], x[1]]; }
    }
  }

  step() {
    if (this.done) return;
    const { n, f, rabbit } = this;
    const [lb, ub] = BOUNDS;
    const mean = [0, 1].map((d) => this.X.reduce((s, x) => s + x[d], 0) / n);
    let Esum = 0;

    for (let i = 0; i < n; i++) {
      const X = this.X[i];
      const E0 = 2 * Math.random() - 1;
      const E = 2 * E0 * (1 - this.t / this.T);
      const absE = Math.abs(E);
      Esum += absE;
      let next;

      if (absE >= 1) {
        if (Math.random() >= 0.5) {
          const Xr = this.X[Math.floor(Math.random() * n)];
          const r1 = Math.random(), r2 = Math.random();
          next = X.map((_, d) => Xr[d] - r1 * Math.abs(Xr[d] - 2 * r2 * X[d]));
          this.strategy[i] = STRATEGY.PERCH_RANDOM;
        } else {
          const r3 = Math.random(), r4 = Math.random();
          next = X.map((_, d) => (rabbit[d] - mean[d]) - r3 * (lb + r4 * (ub - lb)));
          this.strategy[i] = STRATEGY.PERCH_MEAN;
        }
      } else {
        const r = Math.random();
        const J = 2 * (1 - Math.random()); // rabbit's random jump strength
        if (r >= 0.5 && absE >= 0.5) {
          next = X.map((_, d) => (rabbit[d] - X[d]) - E * Math.abs(J * rabbit[d] - X[d]));
          this.strategy[i] = STRATEGY.SOFT;
        } else if (r >= 0.5) {
          next = X.map((_, d) => rabbit[d] - E * Math.abs(rabbit[d] - X[d]));
          this.strategy[i] = STRATEGY.HARD;
        } else {
          const ref = absE >= 0.5 ? X : mean;
          const Y = X.map((_, d) => rabbit[d] - E * Math.abs(J * rabbit[d] - ref[d])).map(clamp);
          const Z = Y.map((y) => y + Math.random() * levy()).map(clamp);
          const fX = f(X[0], X[1]), fY = f(Y[0], Y[1]), fZ = f(Z[0], Z[1]);
          next = fY < fX ? Y : fZ < fX ? Z : X; // greedy: only move if it improves
          this.strategy[i] = absE >= 0.5 ? STRATEGY.SOFT_DIVE : STRATEGY.HARD_DIVE;
        }
      }
      this.X[i] = next.map(clamp);
    }

    this.E = Esum / n;
    this.t++;
    this.Emax = 2 * (1 - this.t / this.T); // envelope that |E| can reach
    this._updateRabbit();
    this.history.push(this.rabbitFit);
  }
}
