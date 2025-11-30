```bash
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3 //何ならここからやる必要あるっぽい。
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc //これやんないと(base)って出ませんのでご注意を〜。
conda —-version
conda create -n labenv python=3.10
conda activate labenv
```