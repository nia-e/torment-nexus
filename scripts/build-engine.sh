#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
revision=922be44aa6ac81b46f092716351cddff1c1733a7
if [ ! -d vendor/llama.cpp/.git ]; then
    mkdir -p vendor
    git clone --filter=blob:none --no-checkout https://github.com/PrismML-Eng/llama.cpp.git vendor/llama.cpp
    git -C vendor/llama.cpp checkout --detach "$revision"
fi
test "$(git -C vendor/llama.cpp rev-parse HEAD)" = "$revision" || {
    echo 'Refusing to build: engine revision differs from the pinned Prism revision.' >&2
    exit 1
}
test -z "$(git -C vendor/llama.cpp status --porcelain --untracked-files=no)" || {
    echo 'Refusing to build: pinned engine has tracked modifications.' >&2
    exit 1
}
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build --target torment-engine -j "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
