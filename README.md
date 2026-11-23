# CSE422_project
As title

# Guide
## 1. Install depenencies
Install [Python](https://www.python.org/downloads/) from the offical site.

And then start a virtual environment:
Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Linux:
```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2. Convert the video to matrix
```bash
python video_to_matrix.py
```

## 3. Run RPCA
```bash
python experiment_rpca.py test1
```
Substitute `test1` with the test you want to use.