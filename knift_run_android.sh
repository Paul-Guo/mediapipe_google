# Switch to OpenCV 4
sed -i -e 's:3.4.3/opencv-3.4.3:4.0.1/opencv-4.0.1:g' WORKSPACE
sed -i -e 's:libopencv_java3:libopencv_java4:g' third_party/opencv_android.BUILD

# Build and install app
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu
rm -f ~/Downloads/templatematchingcpu.apk
cp bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu/templatematchingcpu.apk ~/Downloads

# Switch back to OpenCV 3
sed -i -e 's:4.0.1/opencv-4.0.1:3.4.3/opencv-3.4.3:g' WORKSPACE
sed -i -e 's:libopencv_java4:libopencv_java3:g' third_party/opencv_android.BUILD
rm -f WORKSPACE-e
rm -f third_party/opencv_android.BUILD-e

# install
adb install -r ~/Downloads/templatematchingcpu.apk
