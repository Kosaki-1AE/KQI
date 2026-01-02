```bash
sudo groupadd smbshare
sudo usermod -aG smbshare sasaki
sudo mkdir -p /srv/share
sudo chown -R Society5:smbshare /srv/share
sudo chmod -R 2775 /srv/share
```