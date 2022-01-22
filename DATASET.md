# Preparing Data

1. Data Downloading

    1. Download the AVA raw frames (following [SlowFast](https://github.com/facebookresearch/SlowFast)). 

        ```
        ./script/download_AVA_dataset.sh
        ```

    2. Annotations

        The annotations are contained in [Pasta.tar.gz](https://sjtueducn-my.sharepoint.com/:u:/g/personal/douyiming_sjtu_edu_cn/ETA7mjyrIM1DmVNIBRoY6tcB4KOM98hOv2Rr5TpFMvbk9w?e=dmVOlh).

        Please download and extract it, and then set the AVA.ANNOTATION_DIR to the extracted folder.

        ```
        tar xzvf Pasta.tar.gz
        ```

        Finally, the structure of the downloaded data should be like this:

        ```
        ava_annotations
        |_ annotation
        |	|_ video_pasta_train.csv
        |	|_ video_pasta_val.csv
        |_ frames
        |  |_ [video name 0]
        |  |  |_ [video name 0]_000001.jpg
        |  |  |_ [video name 0]_000002.jpg
        |  |  |_ ...
        |  |_ [video name 1]
        |     |_ [video name 1]_000001.jpg
        |     |_ [video name 1]_000002.jpg
        |     |_ ...
        |_ frame_lists
        |  |_ train.csv
        |  |_ val.csv
        |_ misc
        |	|_ ava_action_list_v2.2_for_activitynet_2019.pbtxt
        |	|_ ava_val_excluded_timestamps_v2.2.csv
        |	|_ ava_val_v2.2.csv
        |_ part_state.pbtxt

        ```

2. Annotation Data Format

    1. annotation

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
        - action: the action label of the person in the human box
        - human_id: ID of the person performing the action
        - foot, leg, etc. : the part states of each part
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

        

