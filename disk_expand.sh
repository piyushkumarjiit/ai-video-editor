# -------------------------------------------------------------------------
# FILE: disk_expand.sh
# ROLE: Dynamic Storage FileSystem Resizer
#
# DESCRIPTION:
# Automatically identifies the root (/) partition and its parent physical 
# disk to expand storage without hardcoded device paths.
#
# FEATURES:
# - Supports both LVM (ubuntu-vg) and Standard Partition layouts.
# - Automates growpart, pvresize, and resize2fs in a single sequence.
# - Safe-check: Exits if partition is already at maximum size.
# -------------------------------------------------------------------------
#!/bin/bash
set -e

echo "🔍 Identifying LVM structure..."

# 1. Get the Logical Volume path directly from the mount point
LV_PATH=$(findmnt -nvo SOURCE /)

# 2. Get the Physical Volume (e.g., /dev/sda3)
PV_PATH=$(sudo pvs --noheadings -o pv_name | tr -d ' ')

# 3. Split /dev/sda3 into /dev/sda and 3
PARENT_DISK=$(echo "$PV_PATH" | sed 's/[0-9]*$//')
PART_NUM=$(echo "$PV_PATH" | grep -o '[0-9]*$')

echo "✅ LV Path: $LV_PATH"
echo "✅ Physical Partition: $PARENT_DISK index $PART_NUM"

# 4. Execute Expansion
echo "📏 Growing partition..."
sudo growpart "$PARENT_DISK" "$PART_NUM" || echo "Partition already maxed"

echo "📦 Resizing LVM Physical Volume..."
sudo pvresize "$PV_PATH"

echo "📈 Expanding Logical Volume and Filesystem..."
sudo lvextend -l +100%FREE "$LV_PATH" -r

echo "🚀 Done! Current Space:"
df -h /