import os
import Tree


class FileManager:

    def __init__(self, source_tree_dir, word_embedding=False):
        self.path_dir = source_tree_dir
        file_list = os.listdir(source_tree_dir)
        self.all_node_list = list()
        self.all_vector_list = list()
        self.answer_list = list()
        for file in file_list:
            if file == "answer":
                continue
            full_name = os.path.join(source_tree_dir, file)  # 파일의 풀네임 == 디렉토리/파일.txt
            text_file = open(full_name, 'r')  # 파일객체
            node_list = text_file.read().splitlines()
            file_name = os.path.basename(full_name)
            node = Tree.Node(file_name, node_list)
            self.all_vector_list.append(node_list)
            if word_embedding is False:
                answer_file_path = os.path.join(source_tree_dir + "/answer", node.file_name + "_bug.txt")
                if os.path.isfile(answer_file_path) is False:
                    continue
                answer_file = open(answer_file_path, 'r')
                node.is_buggy = answer_file.readline()
                self.answer_list.append(answer_file.readline())
            self.all_node_list.append(node)
