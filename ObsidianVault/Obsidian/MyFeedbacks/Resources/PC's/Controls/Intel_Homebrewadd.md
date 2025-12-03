```bash
# 0) Homebrewを入れる
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 0-1) Homebrewが見えない場合ね
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile 
eval "$(/usr/local/bin/brew shellenv)"

# 0-2) Python 3.12 を入れる
brew install python@3.12

# 1) py コマンドを 3.12 に固定
alias py="/usr/local/opt/python@3.12/bin/python3" ⇄ unalias py 2>/dev/null
mkdir -p ~/bin ⇄ rm -f ~/bin/py
ln -s /usr/local/opt/python@3.12/bin/python3.12 ~/bin/py
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
hash -r
py --version   # ← Python 3.12.x
py -m pip install --upgrade pip
pip install "qiskit>=1.0" qiskit-aer

# 1-1) なんかミスって仮想環境を 3.12 で作り直す場合
deactivate 2>/dev/null
rm -rf labenv
/usr/local/opt/python@3.12/bin/python3.12 -m venv labenv
source labenv/bin/activate
python --version
python -m pip install --upgrade pip
pip install "qiskit>=1.0" qiskit-aer
```