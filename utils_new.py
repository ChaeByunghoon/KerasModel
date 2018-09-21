import Tree as tr
import numpy as np
import os
import tensorflow as tf
import flag


class BatchManager:
    def __init__(self, x, y, batch_size, valid_ratio=0.1, rand=True):
        self.train_x, self.train_y = None, None
        self.valid_x, self.valid_y = None, None
        if len(x) != len(y):
            raise ValueError("입력 x와 y의 갯수가 일치하지 않습니다.")
        self.batch_size = batch_size
        self.is_random = rand
        self.valid_ratio = valid_ratio
        self.non_bug_x = list()
        self.non_bug_y = list()
        self.bug_x = list()
        self.bug_y = list()
        for x_d, y_d in zip(x, y):
            print(y_d)
            if y_d == '1':
                self.bug_x.append(x_d)
                self.bug_y.append(y_d)
            else:
                self.non_bug_x.append(x_d)
                self.non_bug_y.append(y_d)
        self.num_data = len(y)
        self.num_train = int(self.num_data * (1 - valid_ratio))
        self.num_valid = self.num_data - self.num_train

    def cross_setting(self, index):
        max_index = int(1 / self.valid_ratio) - 1
        if index == max_index:
            self.valid_x = self.bug_x[int(index * (self.num_valid / 2)):]
            self.valid_x.extend(self.non_bug_x[int(index * (self.num_valid / 2)):])
            self.valid_y = self.bug_y[int(index * (self.num_valid / 2)):]
            self.valid_y.extend(self.non_bug_y[int(index * (self.num_valid / 2)):])
            self.train_x = self.bug_x[:int(index * (self.num_valid / 2))]
            self.train_x.extend(self.non_bug_x[:int(index * (self.num_valid / 2))])
            self.train_y = self.bug_y[:int(index * (self.num_valid / 2))]
            self.train_y.extend(self.non_bug_y[:int(index * (self.num_valid / 2))])
        else:
            self.valid_x = self.bug_x[int(index * (self.num_valid / 2)):int((index + 1) * (self.num_valid / 2))]
            self.valid_x.extend(
                self.non_bug_x[int(index * (self.num_valid / 2)):int((index + 1) * (self.num_valid / 2))])
            self.valid_y = self.bug_y[int(index * (self.num_valid / 2)):int((index + 1) * (self.num_valid / 2))]
            self.valid_y.extend(
                self.non_bug_y[int(index * (self.num_valid / 2)):int((index + 1) * (self.num_valid / 2))])
            self.train_x = self.bug_x[int((index + 1) * (self.num_valid / 2)):]
            self.train_x.extend(self.non_bug_x[int((index + 1) * (self.num_valid / 2)):])
            self.train_y = self.bug_y[int((index + 1) * (self.num_valid / 2)):]
            self.train_y.extend(self.non_bug_y[int((index + 1) * (self.num_valid / 2)):])


    def get_batches(self):
        batch_x, batch_y = list(), list()
        for item_x, item_y in zip(self.train_x, self.train_y):
            batch_x.append(item_x)
            batch_y.append(item_y)
            if len(batch_x) >= self.batch_size:
                yield batch_x, batch_y
                batch_x.clear()
                batch_y.clear()
        if len(batch_x) > 0:
            yield batch_x, batch_y

    def get_valid_batches(self):
        batch_x, batch_y = list(), list()
        for item_x, item_y in zip(self.valid_x, self.valid_y):
            batch_x.append(item_x)
            batch_y.append(item_y)
            if len(batch_x) >= self.batch_size:
                yield batch_x, batch_y
                batch_x.clear()
                batch_y.clear()
        if len(batch_x) > 0:
            yield batch_x, batch_y


class FileManager:
    def __init__(self, source_tree_dir, word_embadding=False):
        self.path_dir = source_tree_dir
        file_list = os.listdir(source_tree_dir)
        self.root_node_list = list()
        for file in file_list:
            if file == "answer":
                continue
            all_node_list = list()  # 한파일내 모든 노드들 level 단위로 저장하기위한 리스트
            full_name = os.path.join(source_tree_dir, file)  # 파일의 풀네임 == 디렉토리/파일.txt
            text_file = open(full_name, 'r')  # 파일객체
            file_lines = text_file.read().splitlines()
            root_node = tr.Node(level=1, vector=file_lines[0])  # 루트노드생성, 레벨=1
            root_node.file_name = os.path.basename(full_name)[:-4]
            if word_embadding is False:
                answer_file_path = os.path.join(source_tree_dir + "/answer", root_node.file_name + "_bug.txt")
                if os.path.isfile(answer_file_path) is False:
                    continue
                answer_file = open(answer_file_path, 'r')
                root_node.is_buggy = answer_file.readline()
            level1_node_list = list()  # 레벨1노드리스트. 한개만담을꺼임
            level1_node_list.append(root_node)  # 한개만담았음
            all_node_list.append(level1_node_list)  # 모든노드 리스트에 추가
            for line in file_lines[1:]:
                level = line.count("	") + 1
                if len(all_node_list) < level:
                    node = tr.Node(level=level, vector=line.replace("	", ""))
                    node.set_parent(all_node_list[level - 2][-1])  # 상위노드의 맨 최신, 즉 맨 끝노드 획득해서 넣어버리기!
                    new_node_list = list()
                    new_node_list.append(node)
                    all_node_list.append(new_node_list)
                else:
                    node = tr.Node(level=level, vector=line.replace("	", ""))
                    node.set_parent(all_node_list[level - 2][-1])  # 상위노드의 맨 최신, 즉 맨 끝노드 획득해서 넣어버리기!
                    all_node_list[level - 1].append(node)
            self.root_node_list.append(root_node)
            root_node.all_node_list = all_node_list


def convert_node_to_vector_list(root, option, split_add=False, test_func=False):
    result_vector_list = list()

    all_node = root.all_node_list
    for leveled_node in all_node:
        for one_node in leveled_node:
            if len(one_node.child) != 0:
                result_vector_list.append(one_node.vector)
                for child_node in one_node.child:
                    result_vector_list.append(child_node.vector)
                if test_func is True:
                    result_vector_list.append("\n")
                elif split_add is True:
                    result_vector_list.append(np.zeros(flag.total_dim))

    """
    for i in range(0, len(result_vector_list)):
        result_vector_list[i] = parse_node_name(node_str=result_vector_list[i])
    """
    return result_vector_list


def _recursive_for_1option(append_list, node):
    if node.level == 1:
        append_list.append(node.vector)
    if len(node.child) == 0:
        pass
    for child_node in node.child:
        append_list.append(child_node.vector)
        _recursive_for_1option(append_list, child_node)


def parse_node_name(node_str):
    if node_str.find("@") != -1:
        idx = node_str.find("@")
        node_str = node_str[0:idx]
    return node_str
