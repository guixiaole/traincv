import json
import os


def get_output_labeljson():
    """
    获取标注好的json文件。当获取完毕之后，就开始删除所有的json文件。
    :return: 返回的是已经设置好的坐标。
    @gxl  8/25
    """
    json_url = 'openlabel/output/PASCAL_VOC/'
    txt_url = 'lumppospredicate.txt'
    label_jsons = os.listdir(json_url)
    cord = []
    for label_json in label_jsons:
        label_url = json_url + label_json
        label_count = int(label_json[:-5])
        with open(label_url, 'r') as f:
            dict_str = json.loads(f.read())
            dict_str = dict_str['shapes'][0]
            pos = dict_str['points']
            print(pos)
            xmax = round(pos[0][0])
            ymax = round(pos[0][1])
            xmin = round(pos[1][0])
            ymin = round(pos[1][1])
            print(label_count, xmax, ymax, xmin, ymin)
            w = abs(int(xmax) - int(xmin)) + 1
            h = abs(int(ymax) - int(ymin)) + 1
            cord.append((label_count, xmax, ymax, w, h))
    print(cord)
    #  获得所有标注的坐标之后，需要将xml文件进行删除。
    """
    暂时先注释掉。后面再
    for label_json in label_jsons:
        if label_json.endswith(".json"):
            os.remove(os.path.join(json_url, label_json))
    """
    #  把预测的数据进行存储。
    for j in range(0, len(cord)):
        with open(txt_url, "a") as file:
            file.write(str(cord[j][0]) + ";")
            for i in range(1, len(cord[j])):
                file.write(str(cord[j][i]) + ",")
            file.write("\n")
    return cord
