# Preparing Data

1. Raw Frames & Annotations

    1. Download the AVA raw frames (following [SlowFast](https://github.com/facebookresearch/SlowFast)), and set the config AVA.FRAME_DIR to the extracted folder.

        ```
        ./scripts/download_AVA_dataset.sh /your/data/path
        ```

    2. Annotations
        
        The annotations are contained in the following packages:
        
        |Version|URL|
        |---|---|
        |AVA v2.1|[hake_ava_annotations_v2.1.tar.gz](https://1drv.ms/u/s!ArUVoRxpBphYguQTX6aTS-LBQqCIUg?e=HFtcwR)|
        |AVA v2.2|[hake_ava_annotations.tar.gz](https://1drv.ms/u/s!ArUVoRxpBphYguIe6oP4tYRWAwwqLQ?e=CENEMN)|

        Please download and extract it, and set the config AVA.ANNOTATION_DIR to the extracted folder.

        ```
        tar xzvf hake_ava_annotations.tar.gz # add "_v2.1" after "annotations" for AVA v2.1
        ```

        Finally, the structure of the downloaded data should be like this:

        ```
        hake_ava_annotations
        |_ ava_val_predicted_boxes.csv
        |_ video_pasta_train.csv
        |_ video_pasta_val.csv
        |_ train.csv
        |_ val.csv
        |_ ava_action_list_v2.2_for_activitynet_2019.pbtxt
        |_ ava_val_excluded_timestamps_v2.2.csv
        |_ ava_val_v2.2.csv
        |_ part_state.pbtxt

        ```

2. Annotation Format

    1. video_pasta_{train/val}.csv

        These files contain the annotations of each frame, including human boxes, actions, part states, etc.

        example:

        | video       | frame | x1    | y1    | x2    | y2    | action | human_id | foot | leg  | hip  | hand | arm  | head |
        | ----------- | ----- | ----- | ----- | ----- | ----- | ------ | -------- | ---- | ---- | ---- | ---- | ---- | ---- |
        | -5KQ66BBWC4 | 902   | 0.077 | 0.151 | 0.283 | 0.811 | 80     | 1        | -1   | -1   | -1   | 17   | -1   | 1    |
        | -5KQ66BBWC4 | 902   | 0.077 | 0.151 | 0.283 | 0.811 | 9      | 1        | 14   | 13   | -1   | 32   | 6    | -1   |
        | -5KQ66BBWC4 | 902   | 0.226 | 0.032 | 0.366 | 0.497 | 12     | 0        | 0    | -1   | -1   | 17   | -1   | 1    |

        The meanings of each column:

        - video: name of the video
        - frame: time (second) of the frame
        - x1, y1: the upper left corner of human box
        - x2, y2: the bottom right corner of human box
        - action: the action label of the person in the human box (1-based)
        - human_id: ID of the person performing the action
        - foot, leg, etc. : the part states of each part (0-based)
            - **-1** means there's no part state for the corresponding part.

    2. part_state.pbtxt

        This file contains the names and ids of each part state.

        example:

        ```
        item {
          name: "hip: sit on"    # the part name is "hip", and the state is "sit on".
          id: 30                 # the id is 30, which is used in the evaluation stage.
        }
        ```

        

