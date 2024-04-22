
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_tracking_gpu_video_input
bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_gpu_video_input \
  --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_gpu_video_input.pbtxt \
  --input_side_packets=input_video_path=9.mov,output_video_path=out_linux_gpu.mp4
