FROM mcr.microsoft.com/devcontainers/base:bookworm

ARG TZ=UTC
ENV TZ="$TZ" DEVCONTAINER=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git iproute2 iptables jq less procps ruby tmux unzip \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (anonymous S3 access to fcp-indi, openneuro.org).
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscliv2.zip \
  && unzip -q /tmp/awscliv2.zip -d /tmp \
  && /tmp/aws/install \
  && rm -rf /tmp/aws /tmp/awscliv2.zip

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# IBM Aspera CLI, for TCIA releases distributed only as Aspera Faspex packages.
RUN gem install --no-document aspera-cli -v 4.27.3

# Drop the base image's blanket passwordless sudo; allow only the firewall script.
COPY init-firewall.sh /usr/local/bin/
RUN chmod 755 /usr/local/bin/init-firewall.sh \
  && rm -f /etc/sudoers.d/vscode \
  && echo "vscode ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/vscode-firewall \
  && chmod 0440 /etc/sudoers.d/vscode-firewall

# Volume mount points live under $HOME so the devcontainer UID remap re-owns them.
RUN mkdir -p /home/vscode/.claude /home/vscode/.history \
  && chown vscode:vscode /home/vscode/.claude /home/vscode/.history

USER vscode
RUN curl -fsSL https://claude.ai/install.sh | bash
# The ascp transfer binary (IBM Aspera Transfer SDK), installed under ~/.aspera.
RUN ascli config ascp install
ENV PATH="/home/vscode/.local/bin:$PATH"
