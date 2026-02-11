# SSH & File Transfer Guide

Quick reference for connecting to the VM and syncing files.

---

## 1. SSH into the VM

```bash
ssh -i /path/to/your-key.pem user@<VM_IP_ADDRESS>
```

- `-i /path/to/your-key.pem` — path to your private key (certificate file)
- Replace `user` with your VM username (e.g., `root`, `ubuntu`)
- Replace `<VM_IP_ADDRESS>` with the VM's public IP

### First-time connection

You'll see a fingerprint prompt — type `yes` to add the host to known hosts.

### Common SSH flags

```bash
ssh -i key.pem -p 22 user@<VM_IP>       # Specify port (default is 22)
ssh -i key.pem -L 8080:localhost:8080 user@<VM_IP>  # Port forwarding (access VM's port 8080 locally)
```

---

## 2. Sync files with rsync

**rsync** copies files between your local machine and the VM. It only transfers what's changed, making it much faster than copying everything each time.

### Local to VM (upload)

```bash
rsync -avz -e "ssh -i /path/to/your-key.pem" \
  /local/path/to/project/ \
  user@<VM_IP>:/remote/path/to/project/
```

### VM to local (download)

```bash
rsync -avz -e "ssh -i /path/to/your-key.pem" \
  user@<VM_IP>:/remote/path/to/project/ \
  /local/path/to/project/
```

### Flags explained

| Flag | Meaning |
|------|---------|
| `-a` | Archive mode — preserves permissions, timestamps, symlinks |
| `-v` | Verbose — shows which files are being transferred |
| `-z` | Compress during transfer (faster over slow connections) |
| `-e "ssh -i key.pem"` | Use SSH with your certificate file |

### Exclude files from sync

```bash
rsync -avz -e "ssh -i /path/to/your-key.pem" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'indexes/' \
  /local/path/to/project/ \
  user@<VM_IP>:/remote/path/to/project/
```

---

## 3. Copy a single file with scp

For quick one-off file transfers:

```bash
# Upload
scp -i /path/to/your-key.pem localfile.txt user@<VM_IP>:/remote/path/

# Download
scp -i /path/to/your-key.pem user@<VM_IP>:/remote/path/file.txt ./
```

---

## 4. Tips

- **Key permissions**: Your `.pem` file must have restricted permissions or SSH will refuse it:
  ```bash
  chmod 400 /path/to/your-key.pem
  ```
- **Trailing slash matters in rsync**: `project/` syncs the *contents*, `project` syncs the folder itself
- **Dry run**: Add `--dry-run` to rsync to preview what would be transferred without actually doing it
- **Keep SSH alive**: Add to `~/.ssh/config` to prevent timeouts:
  ```
  Host *
      ServerAliveInterval 60
      ServerAliveCountMax 3
  ```
