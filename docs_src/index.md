# pystreamgraph

A vibe-coded package to help with the plotting of streamgraphs in python. Plotting streamgraphs was already possible in matplotlib, but quite messy. This package should make it a one-liner, with a lot of options for layout, smoothing, label placement etc.



## Install

```bash
pip install pystreamgraph
```

### Install from GitHub

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
X_ = np.arange(n)
base = np.linspace(0, 2*np.pi, n)
Y_ = []
for i in range(k):
    phase = rng.uniform(0, 2*np.pi)
    amp = rng.uniform(0.6, 1.3)
    y = amp * (np.sin(base + phase) + 1.2) + rng.normal(0, 0.08, size=n) + 0.15
    y = np.clip(y, 0, None)
    Y_.append(y)
Y_ = np.vstack(Y_)

ax = plot_streamgraph(X_, Y_, labels=list("ABCDE"), sorted_streams=False,
                      margin_frac=0.10, smooth_window=3, cmap='magma',
                      curve_samples=40,alpha=0.9,label_color='grey',label_placement=True,label_position='max_width')
ax.set_title("Streamgraph")
plt.show()
```

![Example streamgraph](images/streamgraph_base.png)



## Links

- Docs: https://MNoichl.github.io/pystreamgraph/
- Source code: https://github.com/MNoichl/pystreamgraph
- Issues: https://github.com/MNoichl/pystreamgraph/issues


## Inspiration

This package takes up ideas from these papers, among others: 

Byron, L., & Wattenberg, M. (2008). Stacked graphs—Geometry & aesthetics. IEEE Transactions on Visualization and Computer Graphics, 14(6), 1245–1252. [https://doi.org/10.1109/TVCG.2008.166](https://doi.org/10.1109/TVCG.2008.166)

Di Bartolomeo, M., & Hu, Y. (2016). There is more to streamgraphs than movies: Better aesthetics via ordering and lassoing. Computer Graphics Forum, 35(3), 341–350. [https://doi.org/10.1111/cgf.12910](https://doi.org/10.1111/cgf.12910)

Havre, S., Hetzler, B., & Nowell, L. (2000). ThemeRiver: Visualizing theme changes over time. In IEEE Symposium on Information Visualization (InfoVis 2000) (pp. 115–123). IEEE. [https://doi.org/10.1109/INFVIS.2000.885098](https://doi.org/10.1109/INFVIS.2000.885098)