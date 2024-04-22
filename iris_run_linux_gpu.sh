
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_tracking_gpu
bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_gpu \
  --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_gpu.pbtxt \
  --input_video_path=7.mov \
  --output_video_path=out_linux_gpu.mp4
