import os


class SerializationProcess:
    def __init__(self, default_path="./temp/serializations"):
        self.serialization_str = ""
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        self.save_path = default_path

    def adding_sequence(self, data):
        self.serialization_str += data

    def save_file(self, name):
        # 目前定义为 template 和 predict 两种类型，最终生成文件也是这两个文件
        print("Save")
        with open(os.path.join(self.save_path, name), "w") as txt_file:
            txt_file.write(self.serialization_str.strip('0'))
        self.serialization_str = ""
