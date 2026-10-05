"""Execute saved-result display cells and publish a notebook with real outputs.

The displayed data/logs/figures come from the completed local study. Training
branches remain runnable on Colab. The display run performs no model inference.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def wrap(original, section):
    return ('if globals().get("VIEW_SAVED_RESULTS", False):\n'
            f'    results.show("{section}")\nelse:\n' + textwrap.indent(original, "    "))


def build(reference):
    import build_notebook
    # Generate the clean training template in memory; don't erase the published
    # notebook's outputs as a side effect of exporting/verification.
    notebook = nbformat.from_dict(build_notebook.build(write=False))
    notebook.cells[0].source = """# DeepWeeds — Lab Day 2: kết quả thực nghiệm và chạy lại trên Colab

**Notebook đã có output từ nghiên cứu local thật**: GTX1650, profile hour, 128px,
1epoch khảo sát và 12epoch chung kết, 3 seed. Log/CSV/PNG gốc nằm trong bài nộp.
Các cell ở chế độ mặc định chỉ đọc và hiển thị kết quả đã kiểm tra hash; không train/test lại.
Số thứ tự thực thi trong bản này là lần chạy các cell hiển thị, không phải một phiên train mới.

- Xem ngay các bảng, đồ thị/ảnh và log bên dưới trên GitHub hoặc Colab.
- Muốn train lại: đặt **VIEW_SAVED_RESULTS=False**, chọn GPU, dùng SESSION mới và Run all.
- PROFILE=hour: 128px, 1epoch khảo sát; FINAL_EPOCHS=12 như lượt local. Batch/AMP chọn
  theo GPU được cấp, nên thời gian và kết quả có thể khác máy local.
- Temperature và lựa chọn chỉ trên val; test cuối cùng một lần mỗi cấu hình/seed.

[Báo cáo hoàn thiện](https://github.com/meth04/K4-DAY02-NguyenVanThan-02859/blob/main/submissions/02859_nguyen_van_than/report.md)
· [Bài nộp](https://github.com/meth04/K4-DAY02-NguyenVanThan-02859/tree/main/submissions/02859_nguyen_van_than)
"""
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 12
    code[0].source = '''VIEW_SAVED_RESULTS = True      # True: xem kết quả đã lưu; False: train lại trên Colab
PROFILE = "hour"               # hour | fast | full | smoke
FINAL_EPOCHS = 12               # Lượt local đã train 12 epoch chung kết
SAVE_TO_DRIVE = True
SESSION = "deepweeds_day2_reproduction"  # SESSION mới khi train nghiên cứu mới
ROOT = f"/content/{SESSION}_{PROFILE}"
print("Chế độ:", "xem kết quả đã lưu, không train/test lại" if VIEW_SAVED_RESULTS else "train mới trên Colab")
print("Profile chạy lại:", PROFILE, "| final epochs:", FINAL_EPOCHS)
'''
    original_setup = code[1].source
    code[1].source = '''import os, sys, json
from pathlib import Path
if VIEW_SAVED_RESULTS:
    import importlib.util, subprocess
    if importlib.util.find_spec("openpyxl") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openpyxl"], check=True)
    import pandas as pd
    from IPython.display import display
    print("Chế độ xem kết quả: không cần GPU. Phần cứng train thật được hiển thị ở cell khởi tạo.")
else:
''' + textwrap.indent(original_setup, "    ")
    original_download = code[2].source
    code[2].source = f'''from urllib.request import urlretrieve
from zipfile import ZipFile
if VIEW_SAVED_RESULTS:
    RESULTS_COMMIT = "{reference}"
    local_results = os.environ.get("DEEPWEEDS_SAVED_RESULTS_DIR")
    if local_results:  # Chỉ dành cho xuất notebook offline trên máy có artifact
        saved_root = Path(local_results)
        helper = Path(os.environ["DEEPWEEDS_RESULTS_HELPER"])
    else:
        cache = Path("/content" if Path("/content").exists() else ".") / f"deepweeds_saved_{{RESULTS_COMMIT[:7]}}"
        cache.mkdir(parents=True, exist_ok=True)
        saved_root = cache / "submission"
        helper = cache / "notebook_results.py"
        if not (saved_root / "manifest.json").exists() or not helper.exists():
            archive_path = cache / "source.zip"
            urlretrieve(f"https://codeload.github.com/meth04/K4-DAY02-NguyenVanThan-02859/zip/{{RESULTS_COMMIT}}", archive_path)
            with ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    relative = member.partition("/")[2]
                    prefix = "submissions/02859_nguyen_van_than/"
                    if relative.startswith(prefix) and not member.endswith("/"):
                        target = saved_root / relative[len(prefix):]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
                    elif relative == "code/notebook_results.py":
                        helper.write_bytes(archive.read(member))
    sys.path.insert(0, str(helper.parent))
    from notebook_results import SavedResults
    results = SavedResults(saved_root)
    results.show("source")
else:
''' + textwrap.indent(original_download, "    ")
    initialize = code[3].source.replace('lab = FastLab(ROOT, PROFILE, backup=BACKUP, settings=settings)',
        'lab = FastLab(ROOT, PROFILE, backup=BACKUP, settings=settings)\nlab.budget["final"] = FINAL_EPOCHS')
    code[3].source = wrap(initialize, "initialize")
    sections = ["data","checks","backbones","ablations","inference","finals","products","download"]
    for cell, section in zip(code[4:], sections):
        cell.source = wrap(cell.source, section)
    # Show the original training environment immediately after artifacts load.
    code[2].source = code[2].source.replace('results.show("source")', 'results.show("source")\n    results.show("environment")')
    # Keep training headings accurate for the current saved-results run.
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        if cell.source.startswith("## 6."):
            cell.source = "## 6. So sánh 5 backbone\n\nOutput gốc: cùng fold0/seed0/128px/batch64/FP32, 1epoch. Khi train lại, batch/precision chọn theo GPU."
        elif cell.source.startswith("## 8."):
            cell.source = "## 8. Chung kết seed0 và lựa chọn suy luận trên val\n\nLog train gốc và bảng inference. Bảng có đo latency ensemble bổ sung; ghi rõ forward-only gốc so với đo forward+gộp+softmax bổ sung."
        elif cell.source.startswith("## 10."):
            cell.source = "## 10. Kết quả test, phân tích lỗi và workbook\n\nLượt train tự sinh workbook 7 sheet; bài nộp hoàn thiện có 8 sheet, thêm Robustness. Dữ liệu test được đọc từ CSV cũ; không chạy model/test lại."
        elif cell.source.startswith("## 11."):
            cell.source = "## 11. Tải sản phẩm\n\nChế độ xem: tải notebook có output hoặc mở bài nộp trên GitHub. Chế độ train: tải ZIP mới sau Run all; checkpoint lưu Drive."
    index = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "markdown" and cell.source.startswith("## 11."))
    notebook.cells[index:index] = [nbformat.v4.new_markdown_cell("## 10.1. Minh chứng bổ sung: lệch phân phối và Grad-CAM trên validation"),
        nbformat.v4.new_code_cell('if globals().get("VIEW_SAVED_RESULTS", False):\n    results.show("extras")\nelse:\n    print("Bonus đã đo riêng trên validation của nghiên cứu gốc; xem report.md/code/submission_extras.py.")\n')]
    notebook.metadata["saved_results"] = dict(artifact_commit=reference,
        training_commit="9747314cb5aa60fc727fefaa089aacc3de544d15",
        mode="executed artifact display; no training or test inference", profile="hour", final_epochs=12)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True, help="Published commit containing artifacts and notebook_results.py")
    parser.add_argument("--submission", type=Path, default=REPO / "submissions/02859_nguyen_van_than")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_ref):
        parser.error("source-ref must be a full published commit SHA")
    subprocess.run(["git", "cat-file", "-e", f"{args.source_ref}:code/notebook_results.py"],
                   cwd=REPO, check=True, capture_output=True)
    notebook = build(args.source_ref)
    environment = dict(os.environ, DEEPWEEDS_SAVED_RESULTS_DIR=str(args.submission.resolve()),
                       DEEPWEEDS_RESULTS_HELPER=str(REPO / "code/notebook_results.py"),
                       PYTHONDONTWRITEBYTECODE="1", IPYTHONDIR=str(REPO / "_smoke/notebook_ipython"),
                       JUPYTER_RUNTIME_DIR=str(REPO / "_smoke/notebook_jupyter"))
    # This is a real Jupyter execution of artifact display cells. No output is
    # fabricated or pasted into cells that pretend to have trained models.
    client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO)}})
    client.execute(env=environment)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None and cell.outputs for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    images = sum("image/png" in output.get("data", {}) for cell in cells for output in cell.outputs)
    assert images >= 20
    path = REPO / "code/lab_day2.ipynb"
    nbformat.write(notebook, path)
    print(f"Saved {path}: {len(cells)} executed display cells, {images} inline PNG outputs, {path.stat().st_size:,} bytes.")


if __name__ == "__main__":
    main()
