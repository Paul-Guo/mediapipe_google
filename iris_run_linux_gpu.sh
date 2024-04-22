
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/iris_tracking:iris_tracking_gpu_video_input
bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_gpu_video_input \
  --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_gpu_video_input.pbtxt \
  --input_side_packets=input_video_path=9.mov,output_video_path=out_linux_gpu_centos.mp4
