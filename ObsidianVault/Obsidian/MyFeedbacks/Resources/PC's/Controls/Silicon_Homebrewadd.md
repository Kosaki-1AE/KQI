```bash
# 0) Homebrew（未導入なら）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 0-1) Homebrewが見えない場合ね
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 0-2) Python 3.12 を入れる
brew install python@3.12

# 1) py コマンドを 3.12 に固定（古い alias/symlink を掃除してから）
unalias py 2>/dev/null 　
rm -f ~/bin/py ⇄ mkdir -p ~/bin
ln -s /opt/homebrew/opt/python@3.12/bin/python3.12 ~/bin/py
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
hash -r  #　キャッシュ掃除
py --version   # ← Python 3.12.x が出ればOK

# 1-1) なんかミスって仮想環境を 3.12 ベースで作り直す場合（任意のプロジェクト直下で）
deactivate 2>/dev/null
rm -rf labenv
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv labenv
source labenv/bin/activate
python --version    # ← 3.12.x
python -m pip install --upgrade pip
pip install "qiskit>=1.0" qiskit-aer
```