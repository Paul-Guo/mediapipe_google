bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/iristrackinggpu:iristrackinggpu
adb install -r bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/iristrackinggpu/iristrackinggpu.apk
