cat > buildozer.spec << 'EOF'
[app]
title = YOLO Detection
package.name = yoloapp
package.domain = org.yolo

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx

version = 1.0
requirements = python3,kivy==2.1.0,kivymd==1.1.1,numpy

android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.api = 28
android.minapi = 21
android.archs = armeabi-v7a

orientation = landscape

[buildozer]
log_level = 2
EOF
# Build trigger
