bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
mediapipe/examples/desktop/template_matching:template_matching_tflite
bazel-bin/mediapipe/examples/desktop/template_matching/template_matching_tflite \
--calculator_graph_config_file=mediapipe/graphs/template_matching/index_building.pbtxt \
--input_side_packets="file_directory=/Users/guohongcheng/Downloads/mediapipe-master/bounding_box_imgs,file_suffix=png,output_index_filename=mediapipe/models/knift_plu_index.pb"
rm -f mediapipe/models/knift_plu_labelmap.txt
python knift_run_image_model_map_gen.py

echo 'mediapipe/models/knift_plu_index.pb 256'
shasum -a 256 mediapipe/models/knift_plu_index.pb

