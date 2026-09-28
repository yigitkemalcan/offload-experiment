#!/bin/bash
# Swap for the offload experiment: a dedicated swapfile on the NVMe RAID0 (ext4). Usage: setup_swap.sh on|off
set -euo pipefail
SWAPFILE=/mnt/raid0/offload-experiment.swap
case "${1:-on}" in
  on)
    if ! grep -q "$SWAPFILE" /proc/swaps; then
      [ -f "$SWAPFILE" ] || { fallocate -l 16G "$SWAPFILE"; chmod 600 "$SWAPFILE"; mkswap "$SWAPFILE"; }
      swapon "$SWAPFILE"
    fi
    cat /proc/swaps ;;
  off)
    swapoff "$SWAPFILE" 2>/dev/null || true
    rm -f "$SWAPFILE"
    cat /proc/swaps ;;
esac
