```powershell
# パスを
$Path = "C:\share"

# 倉庫フォルダ作成
New-Item -ItemType Directory -Force -Path C:\share | Out-Null

# 既存の共有削除
net share warehouse /delete
net share share /delete

# 共有作成(キャッシュ無効)
net share share=C:\share /CACHE:None

# Everyoneに書き込み権限付与
icacls C:\share /grant Everyone:(OI)(CI)M /T

# ファイアウォール開放
netsh advfirewall firewall set rule group="File and Printer Sharing" new enable=Yes
netsh advfirewall firewall set rule group="ファイルとプリンターの共有" new enable=Yes

# 共有一覧確認
net share

# 自分のPC名確認
hostname

# 自分のIP確認
ipconfig

# 不要なユーザー削除
net user <ユーザー名> /delete

# 現在のユーザー一覧
net user
```