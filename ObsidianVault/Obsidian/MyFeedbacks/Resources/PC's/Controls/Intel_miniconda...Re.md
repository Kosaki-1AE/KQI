```bash
open Miniconda3-latest-MacOSX-x86_64.pkg
source ~/.zshrc
bash Miniconda3-latest-MacOSX-x86_64.sh   //condaの大元がこれね
python3 -m venv labenv
conda create -n labenv python=3.12
source labenv/bin/activate
python -m pip install -U pip setuptools wheel
pip install "qiskit>=1.0" qiskit-aer
```
