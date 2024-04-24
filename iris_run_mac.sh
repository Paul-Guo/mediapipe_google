
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_tracking_cpu_video_input
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu_video_input \
  --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu_video_input.pbtxt \
  --input_side_packets=input_video_path=7.mov,output_video_path=out_macos.mp4
