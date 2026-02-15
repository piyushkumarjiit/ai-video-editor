#!/bin/bash

# Patch CUDA math_functions.h to add noexcept specifications
# This resolves conflicts with glibc 2.40's noexcept declarations

echo "Backing up and patching CUDA math_functions.h..."

MATH_HEADER="/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h"
BACKUP="${MATH_HEADER}.backup"

# Backup if not already done
if [ ! -f "$BACKUP" ]; then
    sudo cp "$MATH_HEADER" "$BACKUP"
    echo "✓ Backed up original to $BACKUP"
fi

# Apply patches to add noexcept specifications using the actual patterns from the file
sudo sed -i 's/double                 cospi(double x);/double                 cospi(double x) noexcept;/g' "$MATH_HEADER"
sudo sed -i 's/double                 sinpi(double x);/double                 sinpi(double x) noexcept;/g' "$MATH_HEADER"
sudo sed -i 's/double                 rsqrt(double a);/double                 rsqrt(double a) noexcept;/g' "$MATH_HEADER"
sudo sed -i 's/float                  cospif(float x);/float                  cospif(float x) noexcept;/g' "$MATH_HEADER"
sudo sed -i 's/float                  sinpif(float x);/float                  sinpif(float x) noexcept;/g' "$MATH_HEADER"
sudo sed -i 's/float                  rsqrtf(float a);/float                  rsqrtf(float a) noexcept;/g' "$MATH_HEADER"

echo "✓ Patched $MATH_HEADER"
echo ""
echo "Verifying patches..."
grep -n "cospi.*noexcept" "$MATH_HEADER" | head -2
echo ""
echo "To restore original: sudo cp $BACKUP $MATH_HEADER"
