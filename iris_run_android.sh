bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/iristrackinggpu:iristrackinggpu
rm -f ~/Downloads/iristrackinggpu.apk
cp bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/iristrackinggpu/iristrackinggpu.apk ~/Downloads
adb install -r ~/Downloads/iristrackinggpu.apk
