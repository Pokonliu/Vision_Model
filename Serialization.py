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
        with open(os.path.join(self.save_path, name), "w") as txt_file:
            txt_file.write(self.serialization_str)
        self.serialization_str = ""

    # TODO compare后期应该为static method
    @staticmethod
    def compare(root_path, src_file_name, target_file_name):
        error_index = []
        try:
            with open(os.path.join(root_path, src_file_name), "r") as template:
                src_data = template.read()
            with open(os.path.join(root_path, target_file_name), "r") as template:
                target_data = template.read()
        except Exception as error:
            return False, error_index, "读取文件错误, %s" % error
        if len(src_data) != len(target_data):
            return False, error_index, "两次编制针数不一致"
        for i in range(len(src_data)):
            if src_data[i] != target_data[i]:
                error_index.append(i)
        ratio = (len(src_data) - len(error_index)) * 100 // len(src_data)
        return True, error_index, "%d%%的正确率" % ratio


if __name__ == '__main__':
    a, b, c = SerializationProcess.compare(r".\temp\serializations", "spin001.txt", "Template.txt")
    print(c)
