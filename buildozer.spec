[app]
title = YOLO Detection
package.name = yoloapp
package.domain = org.yolo

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx

version = 1.0
requirements = python3,kivy==2.1.0

android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.api = 28
android.minapi = 21
android.archs = armeabi-v7a

orientation = portrait

[buildozer]
log_level = 2
