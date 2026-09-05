# Reproducible build and test environment for gz_ecm_battery_plugin.
#
#   docker build -t gz_ecm_battery_plugin .
#   docker run --rm gz_ecm_battery_plugin              # unit tests
#   docker run --rm gz_ecm_battery_plugin e2e          # end-to-end test
#
# Pinned to Ubuntu 24.04 + Gazebo Harmonic, the combination the results in the
# accompanying paper were produced on.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl gnupg lsb-release ca-certificates \
        libgtest-dev python3 python3-pip \
        cppzmq-dev libzmq3-dev \
    && curl https://packages.osrfoundation.org/gazebo.gpg --output \
         /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
         > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/plugin
COPY . .

RUN cmake -S . -B build && cmake --build build -j"$(nproc)"

# gz-transport uses multicast discovery, which does not work reliably inside a
# container; pin it to loopback with a fixed partition (same fix documented in
# the README for multi-interface hosts).
ENV GZ_IP=127.0.0.1 \
    GZ_PARTITION=docker_run \
    GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/plugin/build

# Written with printf rather than a heredoc so the image builds with any
# Docker version, not only BuildKit.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'case "${1:-unit}" in' \
    '  unit) ctest --test-dir /opt/plugin/build --output-on-failure ;;' \
    '  e2e)  python3 /opt/plugin/test/test_end_to_end.py ;;' \
    '  all)  ctest --test-dir /opt/plugin/build --output-on-failure && python3 /opt/plugin/test/test_end_to_end.py ;;' \
    '  *)    exec "$@" ;;' \
    'esac' \
    > /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["unit"]
