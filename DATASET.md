# Preparing Dataset for DIO

1. Dataset downloading steps

    1. Download AVA Dataset (following [SlowFast](https://github.com/facebookresearch/SlowFast)). 

        ```
        ./script/download_AVA_dataset.sh
        ```

    2. Downloading annotation

        The annotation is contained in [DIO.tar.gz](https://sjtueducn-my.sharepoint.com/:u:/g/personal/douyiming_sjtu_edu_cn/EVYz3LDK4y9OkmbK1FGJb9YBEGaWiplS56ZrCdXFKZey7A?e=WJ1bvQ)

        Please download it to ava folder and extract data from the package.

    3. Structure of downloaded data

        ```
        DIO
        |_ DIO_annotation
        |  |_ DIO_test.csv
        |  |_ DIO_train.csv
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
        ```

2. Annotation Format

    Files in DIO folder contains the annotations of each frame, including human/object box, action, object name, etc.

    example:

    | video       | frame | h_x1  | h_y1  | h_x2  | h_y2  | o_x1  | o_y1  | o_x2  | o_y2  | action | object_name | human_id | object_id |
    | ----------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ | ----------- | -------- | --------- |
    | -5KQ66BBWC4 | 905   | 0.392 | 0.033 | 0.556 | 0.618 | 0.37  | 0.019 | 0.432 | 0.608 | 6      | stick       | 12       | 0         |
    | -5KQ66BBWC4 | 906   | 0.408 | 0.008 | 0.586 | 0.639 | 0.37  | 0.036 | 0.457 | 0.678 | 6      | stick       | 12       | 0         |
    | -5KQ66BBWC4 | 907   | 0.42  | 0.115 | 0.616 | 0.883 | 0.371 | 0.143 | 0.466 | 0.878 | 6      | stick       | 12       | 0         |

    The meanings of each column:

    - video: name of the video
    - frame: time (second) of the frame
    - h_x1~h_y2: the upper left and bottom right corners of human-box
    - o_x1~o_y2: the upper left and bottom right corners of object-box
    - action: the action label of the person in the human-box
    - object_name: name of object
    - human_id: ID of the person performing the action
    - object_id: category id of the object

