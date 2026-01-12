# Port Configuration Reference

## 🔌 Complete Port Mapping

### Standard Configuration (Updated)

| Service | External Port | Internal Port | Purpose |
|---------|--------------|---------------|---------|
| **RAG Application** | **8000** | 8000 | Main chat interface (Chainlit) |
| **ChromaDB** | **8001** | 8000 | Vector database API |
| **RedisInsight** | **8002** | 8001 | Redis management UI |
| **vLLM Reasoning** | **8005** | 8005 | Reasoning model API (DeepSeek-R1) |
| **vLLM Vision** | **8006** | 8006 | Vision model API (Qwen2.5-VL) |
| **Redis** | **6379** | 6379 | Redis cache |

## 🌐 Access URLs

Once deployed, access services at:

```bash
# Main Application
http://localhost:8000          # RAG Chat Interface

# APIs
http://localhost:8005/v1       # Reasoning API
http://localhost:8006/v1       # Vision API
http://localhost:8001          # ChromaDB API

# Management UIs
http://localhost:8002          # RedisInsight
```

## 🔍 Port Testing

### Test All Services

```bash
# RAG Application
curl http://localhost:8000
# Should return HTML or redirect

# Reasoning API
curl http://localhost:8005/v1/models
# Should return model info JSON

# Vision API
curl http://localhost:8006/v1/models
# Should return model info JSON

# ChromaDB
curl http://localhost:8001/api/v1/heartbeat
# Should return heartbeat response

# Redis
redis-cli -p 6379 ping
# Should return PONG
```

## 🔧 Changing Ports

### In docker-compose.complete.yml

```yaml
services:
  rag_app:
    ports:
      - "8000:8000"  # Change left side: "NEW_PORT:8000"
  
  chromadb:
    ports:
      - "8001:8000"  # Change left side: "NEW_PORT:8000"
  
  redis:
    ports:
      - "6379:6379"  # Standard Redis port
      - "8002:8001"  # Change left side for RedisInsight
  
  vllm_reasoning:
    ports:
      - "8005:8005"  # Change both if needed
  
  vllm_vision:
    ports:
      - "8006:8006"  # Change both if needed
```

### Example: Change RAG App to Port 9000

```yaml
rag_app:
  ports:
    - "9000:8000"  # External:Internal
```

Then access at: `http://localhost:9000`

## 🚨 Port Conflicts

### Check What's Using a Port

```bash
# Linux/Mac
sudo lsof -i :8000

# Or
sudo netstat -tlnp | grep 8000

# Windows
netstat -ano | findstr :8000
```

### Common Conflicts

| Port | Common User | Solution |
|------|-------------|----------|
| 8000 | Django dev server | Change Django to 8080 or change RAG to 9000 |
| 8001 | Other apps | Usually safe, or change to 8003 |
| 8005/8006 | Other vLLM | Stop other vLLM or use different ports |
| 6379 | Existing Redis | Stop other Redis or change to 6380 |

### Quick Fix for Conflicts

```bash
# Stop the conflicting service
sudo systemctl stop redis  # If system Redis conflicts

# Or change ports in docker-compose.complete.yml
ports:
  - "9000:8000"  # Use port 9000 instead
```

## 🔐 Firewall Configuration

### Open Ports (if remote access needed)

```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 8000/tcp  # RAG App
sudo ufw allow 8005/tcp  # Reasoning API
sudo ufw allow 8006/tcp  # Vision API

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=8005/tcp
sudo firewall-cmd --permanent --add-port=8006/tcp
sudo firewall-cmd --reload
```

### Security Note

⚠️ **For production**:
- Only expose RAG app (8000) to users
- Keep APIs (8005, 8006) internal
- Use reverse proxy (nginx) for SSL/TLS
- Implement authentication

## 📊 Port Usage Summary

### Publicly Accessible (User-Facing)
- **8000** - RAG Chat Interface ✅

### API Endpoints (Internal/Backend)
- **8005** - Reasoning API
- **8006** - Vision API
- **8001** - ChromaDB API

### Management (Admin Only)
- **8002** - RedisInsight
- **6379** - Redis

### Recommended Exposure

```
Internet
    ↓
[Nginx Reverse Proxy]
    ↓
Port 8000 (RAG App)
    ↓
Internal Docker Network
    ↓
Ports 8001, 8005, 8006 (APIs)
```

## 🌍 Remote Access Setup

### Access from Another Machine

```bash
# Replace localhost with server IP
http://192.168.1.100:8000  # RAG App
http://192.168.1.100:8005  # Reasoning API
http://192.168.1.100:8006  # Vision API
```

### Configure in docker-compose.complete.yml

```yaml
rag_app:
  command: python -m chainlit run app.py --host 0.0.0.0 --port 8000
  # --host 0.0.0.0 allows external connections
```

Already configured! ✅

## 🔄 Port Mapping Cheat Sheet

### Format
```yaml
ports:
  - "EXTERNAL_PORT:INTERNAL_PORT"
```

### Examples
```yaml
# Standard
- "8000:8000"  # Port 8000 on both host and container

# Different
- "9000:8000"  # Port 9000 on host, 8000 in container

# IP Binding
- "127.0.0.1:8000:8000"  # Only localhost can access
```

## 📱 Mobile/External Testing

### Find Your Server IP

```bash
# Linux/Mac
hostname -I

# Or
ip addr show | grep inet
```

### Access from Phone/Tablet

```
http://YOUR_SERVER_IP:8000
```

Example: `http://192.168.1.50:8000`

## 🐛 Troubleshooting

### Port Not Responding

1. **Check container is running**
```bash
docker compose ps
```

2. **Check port is bound**
```bash
docker port rag_app
# Should show: 8000/tcp -> 0.0.0.0:8000
```

3. **Check logs**
```bash
docker compose logs rag_app
```

4. **Test from inside container**
```bash
docker exec -it rag_app curl localhost:8000
```

### "Address already in use"

```bash
# Find what's using the port
sudo lsof -i :8000

# Kill the process
sudo kill -9 PID

# Or change port in docker-compose.complete.yml
```

## ✅ Verification Checklist

After deployment:

- [ ] RAG App responds: `curl http://localhost:8000`
- [ ] Reasoning API responds: `curl http://localhost:8005/v1/models`
- [ ] Vision API responds: `curl http://localhost:8006/v1/models`
- [ ] ChromaDB responds: `curl http://localhost:8001/api/v1/heartbeat`
- [ ] Redis responds: `redis-cli -p 6379 ping`
- [ ] Can access from browser: `http://localhost:8000`

---

## 📝 Quick Reference

```bash
# Standard ports (after update)
RAG App:     8000
ChromaDB:    8001
RedisUI:     8002
Reasoning:   8005
Vision:      8006
Redis:       6379
```

**Main interface**: http://localhost:8000 ✅
