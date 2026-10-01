#!/bin/bash
# Egress policy: DNS to configured resolvers, HTTP(S) to public addresses only.
# Everything else (private/cluster ranges, the docker host, SSH, ...) is rejected.
set -euo pipefail

iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

for ns in $(awk '/^nameserver/ {print $2}' /etc/resolv.conf); do
  iptables -A OUTPUT -d "$ns" -p udp --dport 53 -j ACCEPT
  iptables -A OUTPUT -d "$ns" -p tcp --dport 53 -j ACCEPT
done

for net in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10 169.254.0.0/16 127.0.0.0/8 224.0.0.0/4; do
  iptables -A OUTPUT -d "$net" -j REJECT
done

iptables -A OUTPUT -p tcp -m multiport --dports 80,443 -j ACCEPT
iptables -A OUTPUT -j REJECT

if command -v ip6tables >/dev/null && ip6tables -L OUTPUT >/dev/null 2>&1; then
  ip6tables -F OUTPUT
  ip6tables -A OUTPUT -o lo -j ACCEPT
  ip6tables -A OUTPUT -j REJECT
fi

echo "firewall: egress limited to public HTTP(S) + DNS"
