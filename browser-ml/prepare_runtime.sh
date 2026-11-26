#!/bin/sh
# Pinned public inference assets; no keys or API calls are required to run the resulting page.
set -eu
cd "$(dirname "$0")"
mkdir -p runtime
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
npm install --prefix "$tmpdir" --ignore-scripts --no-audit --no-fund onnxruntime-web@1.22.0
ort_package="$tmpdir/node_modules/onnxruntime-web"
cp "$ort_package/dist/ort.wasm.min.mjs" "$ort_package/dist/ort-wasm-simd-threaded.mjs" "$ort_package/dist/ort-wasm-simd-threaded.wasm" runtime/
cp "$ort_package/README.md" runtime/onnxruntime-readme.txt
curl -fsSL --retry 3 'https://huggingface.co/Xenova/dinov2-small/resolve/c2bb04a51fab207c420665f1946016107bffc701/onnx/model_quantized.onnx' -o "$tmpdir/dino.onnx"
echo "3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917  $tmpdir/dino.onnx" | shasum -a 256 -c -
cp "$tmpdir/dino.onnx" runtime/dino-int8.onnx
curl -fsSL 'https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.22.0/LICENSE' -o runtime/ONNXRUNTIME-LICENSE.txt
curl -fsSL 'https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.22.0/ThirdPartyNotices.txt' -o runtime/ONNXRUNTIME-NOTICES.txt
curl -fsSL 'https://raw.githubusercontent.com/facebookresearch/dinov2/main/LICENSE' -o runtime/DINOV2-LICENSE.txt
