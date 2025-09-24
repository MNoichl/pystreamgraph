# pystreamgraph

A compact Matplotlib streamgraph helper, partially vibe-coded, with a lot of options for layout, smoothing, label placement etc.

## Install

```bash
pip install pystreamgraph
```

### Install from GitHub (no PyPI name needed)

```bash
pip install git+https://github.com/MNoichl/pystreamgraph.git
```

## Quickstart

```python
import numpy as np
import matplotlib.pyplot as plt
from pystreamgraph import plot_streamgraph

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

![Example streamgraph](https://raw.githubusercontent.com/MNoichl/pystreamgraph/main/images/streamgraph_base.png)

## Links

- Docs: https://MNoichl.github.io/pystreamgraph/
- Source: https://github.com/MNoichl/pystreamgraph
- Issues: https://github.com/MNoichl/pystreamgraph/issues

## API

- `plot_streamgraph(X, Y, ...)` – plot a streamgraph onto a Matplotlib Axes.
- `streamgraph_envelopes(Y, ...)` – compute bottoms/tops per layer.
- `pchip_interpolate(x, y, ...)` – shape‑preserving cubic interpolation.
- `catmull_rom_interpolate(x, y, ...)` – Catmull–Rom curve interpolation.



