bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
mediapipe/examples/desktop/template_matching:template_matching_tflite
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/template_matching/template_matching_tflite \
--calculator_graph_config_file=mediapipe/graphs/template_matching/template_matching_desktop.pbtxt \
--input_side_packets="input_video_path=7.mov,output_video_path=out_knift_macos.mp4"
