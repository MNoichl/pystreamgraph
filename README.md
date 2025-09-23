# streamgraph

A compact Matplotlib streamgraph helper with ordering, margins, optional smoothing, shape‑preserving PCHIP boundary curves (default), optional Catmull–Rom curves, and label placement.

## Install

```bash
pip install streamgraph
```

## Quickstart

```python
import numpy as np
import matplotlib.pyplot as plt
from streamgraph import plot_streamgraph

rng = np.random.default_rng(7)
n, k = 40, 5
X = np.arange(n)
base = np.linspace(0, 2*np.pi, n)
Y = []
for i in range(k):
    phase = rng.uniform(0, 2*np.pi)
    amp = rng.uniform(0.6, 1.3)
    y = amp * (np.sin(base + phase) + 1.2) + rng.normal(0, 0.08, size=n) + 0.15
    y = np.clip(y, 0, None)
    Y.append(y)
Y = np.vstack(Y)

ax = plot_streamgraph(X, Y, labels=list("ABCDE"), sorted_streams=True,
                     margin_frac=0.10, smooth_window=1, cmap=None,
                     curve_samples=16, curve_method="pchip")
ax.set_title("Streamgraph with PCHIP boundaries")
plt.show()
```

![Example streamgraph](images/streamgraph_base.png)


## API

- `plot_streamgraph(X, Y, ...)` – plot a streamgraph onto a Matplotlib Axes.
- `streamgraph_envelopes(Y, ...)` – compute bottoms/tops per layer.
- `pchip_interpolate(x, y, ...)` – shape‑preserving cubic interpolation.
- `catmull_rom_interpolate(x, y, ...)` – Catmull–Rom curve interpolation.

## License

MIT


