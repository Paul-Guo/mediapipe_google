bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu
rm -f ~/Downloads/templatematchingcpu.apk
cp bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu/templatematchingcpu.apk ~/Downloads
adb install -r ~/Downloads/templatematchingcpu.apk
