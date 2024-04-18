bazel build -c opt --config=ios_arm64 mediapipe/examples/ios/iristrackinggpu:IrisTrackingGpuApp
rm -f ~/Downloads/IrisTrackingGpuApp.ipa
cp bazel-bin/mediapipe/examples/ios/iristrackinggpu/IrisTrackingGpuApp.ipa ~/Downloads
open ~/Downloads
