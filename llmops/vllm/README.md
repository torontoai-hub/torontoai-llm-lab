

Allow Firewall

sudo iptables -I INPUT -p tcp --dport 8000 -m state --state NEW,ESTABLISHED -j ACCEPT
sudo iptables -I OUTPUT -p tcp --sport 8000 -m state --state ESTABLISHED -j ACCEPT

Docker forward too

sudo iptables -I FORWARD -p tcp --dport 8000 -j ACCEPT
sudo iptables -I FORWARD -p tcp --sport 8000 -j ACCEPT

curl -s http://<IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role":"user","content":"Say hello in one short sentence."}],
    "temperature": 0.2
  }' | jq

