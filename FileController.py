import os
from Tree import Node
import numpy as np
import random

class FileManager:

    def __init__(self, source_tree_dir):
        self.path_dir = source_tree_dir
        self.answer_path = source_tree_dir + "/answer"
        file_list = os.listdir(source_tree_dir)
        self.all_node_list = list()
        self.all_tree_node_list = list()
        self.answer_list = list()
        file_list.sort()
        for file in file_list:
            if file == "answer" or file == ".DS_Store":
                continue
            full_name = os.path.join(source_tree_dir, file)  # 파일의 풀네임 == 디렉토리/파일.txt
            text_file = open(full_name, 'r', encoding="ascii", errors="surrogateescape")  # 파일객체
            node_list = text_file.read().splitlines()
            self.all_node_list.append(node_list)
            text_file.close()
            answer_file_path = os.path.join(source_tree_dir + "/answer", file[:-4] + "_bug.txt")
            if os.path.isfile(answer_file_path) is False:
                continue
            answer_file = open(answer_file_path, 'r')
            is_buggy = int(answer_file.readline())
            self.all_tree_node_list.append(Node(file, node_list, is_buggy))

        answer_file_list = os.listdir(self.answer_path)
        answer_file_list.sort()
        for answer_file in answer_file_list:
            full_name = os.path.join(self.answer_path, answer_file)
            text_file = open(full_name, 'r', encoding="ascii", errors="surrogateescape")  # 파일객체
            answer = text_file.readline()
            self.answer_list.append(int(answer))
        random.shuffle(self.all_tree_node_list)

    def get_node_answer_list(self):
        node_lists = list()
        answer_list = list()
        for node in self.all_tree_node_list:
            node_lists.append(node.node_list)
            answer_list.append(node.is_buggy)
        return node_lists, answer_list

    def get_train_test_set(self, x_all, y_all):
        bug_list = list()
        bug_answer_list = list()
        non_bug_list = list()
        non_bug_answer_list = list()

        for i in range(0, len(y_all)):
            if y_all[i] == 1:
                bug_list.append(x_all[i])
                bug_answer_list.append(1)
            else:
                non_bug_list.append(x_all[i])
                non_bug_answer_list.append(0)

        train_len = int(len(x_all) * 0.1)
        bug_train_len = int(len(bug_list) * 0.1)
        non_bug_train_len = int(len(non_bug_list) * 0.1)
        new_bug_list, new_bug_answer_list = self._get_copied_list(bug_list[train_len:],
                                                                  len(non_bug_list[train_len:]), bug_answer_list[train_len:])
        x_train = new_bug_list + non_bug_list[train_len:]
        y_train = new_bug_answer_list + non_bug_answer_list[train_len:]

        x_train, y_train = shuffle_data(x_train, y_train)
        x_test = bug_list[:bug_train_len] + non_bug_list[:non_bug_train_len]
        y_test = bug_answer_list[:bug_train_len] + non_bug_answer_list[:non_bug_train_len]
        x_test, y_test = shuffle_data(x_test, y_test)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        print("테스트 셋 버그파일 개수 : ", len(bug_list[:bug_train_len]), "테스트셋 논 버그파일 개수", len(non_bug_list[:non_bug_train_len]))
        return x_train, y_train, x_test, y_test

    def _get_copied_list(self, bug_set, set_num, bug_answer_set):
        new_bug_set = list()
        new_bug_answer_set = list()
        iter_num = int(set_num / len(bug_set))
        for i in range(0, iter_num):
            new_bug_set += bug_set
            new_bug_answer_set += bug_answer_set
        remain_num = set_num - len(new_bug_set)
        for i in range(0, remain_num):
            new_bug_set.append(bug_set[i])
            new_bug_answer_set.append(1)

        return new_bug_set, new_bug_answer_set


def shuffle_data(x, y):
    random_seed = np.random.permutation(len(x))
    shuffle_x = list()
    shuffle_y = list()
    for randIndex in random_seed:
        shuffle_x.append(x[randIndex])
        shuffle_y.append(y[randIndex])
    return shuffle_x, shuffle_y




