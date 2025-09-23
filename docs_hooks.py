from __future__ import annotations
from pathlib import Path
import shutil
import filecmp


def _copy_if_changed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            # Cheap content equality: compare bytes
            if src.read_bytes() == dst.read_bytes():
                return
        except Exception:
            pass
    shutil.copy2(src, dst)


def on_pre_build(config):
    """Prepare docs inputs before build.

    - Copy the demo notebook into docs_src/ as guide.ipynb for mkdocs-jupyter.
    """
    project_root = Path(config.config_file_path).parent
    nb_src = project_root / "notebooks" / "streamgraph_demo.ipynb"
    docs_nb = Path(config.docs_dir) / "guide.ipynb"
    if nb_src.exists():
        _copy_if_changed(nb_src, docs_nb)


def on_post_build(config):
    """Copy project-level images/ into site_dir/images after build to avoid
    touching docs sources during watch (prevents rebuild loops).
    """
    project_root = Path(config.config_file_path).parent
    src_images = project_root / "images"
    site_images = Path(config.site_dir) / "images"

    if not src_images.exists():
        return

    if site_images.exists():
        shutil.rmtree(site_images)
    site_images.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_images, site_images)
