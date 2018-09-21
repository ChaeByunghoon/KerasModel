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

    def except_infrequent_word(self, node_vector_list):

        word_list = list()
        count_list = list()
        new_word_list = list()
        new_node_vector_list = list()

        for node_list in node_vector_list:
            for i in range(0, len(node_list)):
                if node_list[i] in word_list:
                    idx = word_list.index(node_list[i])
                    cnt = count_list[idx] + 1
                    count_list.pop(idx)
                    count_list.insert(idx, cnt)
                else:
                    word_list.append(node_list[i])
                    count_list.append(1)

        for i in range(0, len(count_list)):
            if count_list[i] > 3:
                new_word_list.append(word_list[i])

        for node_list in node_vector_list:
            temp_node_list = list()
            for i in range(0, len(node_list)):
                if node_list[i] in new_word_list:
                    temp_node_list.append(node_list[i])
            new_node_vector_list.append(temp_node_list)

        return new_node_vector_list

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

    def get_train_test_cv_set(self, idx, x_all, y_all):
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

        bug_train_len = int(len(bug_list) * 0.2)
        non_bug_train_len = int(len(non_bug_list) * 0.2)

        # 테스트셋
        x_test = bug_list[(idx - 1)*bug_train_len:idx*bug_train_len] + \
                 non_bug_list[(idx - 1)*non_bug_train_len:idx*non_bug_train_len]
        y_test = bug_answer_list[(idx - 1)*bug_train_len:idx*bug_train_len] +\
                 non_bug_answer_list[(idx - 1)*non_bug_train_len:idx*non_bug_train_len]

        # generate Buggy File
        temp_other_bug_list = bug_list[:(idx - 1)*bug_train_len] + bug_list[idx*bug_train_len:]
        temp_other_answer_list = bug_answer_list[:(idx - 1)*bug_train_len] + bug_answer_list[idx*bug_train_len:]
        copy_num = len(non_bug_list[:(idx-1)*non_bug_train_len]) + len(non_bug_list[idx*non_bug_train_len:])

        new_bug_list, new_bug_answer_list = self._get_copied_list(temp_other_bug_list,
                                                                      copy_num,
                                                                      temp_other_answer_list)
            # 트레이닝 셋 복사된 버그 + 논 버그
        x_train = new_bug_list + non_bug_list[:(idx-1)*non_bug_train_len] + non_bug_list[idx*non_bug_train_len:]
        y_train = new_bug_answer_list + \
                  non_bug_answer_list[:(idx-1)*non_bug_train_len] + non_bug_answer_list[idx*non_bug_train_len:]

        #x_train, y_train = shuffle_data(x_train, y_train)
        #x_test, y_test = shuffle_data(x_test, y_test)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = np.array(y_test)
        y_test = y_test.reshape(len(y_test), 1)
        print("트레이닝셋 파일 개수 : ", len(x_train), "테스트셋 파일 개수", len(x_test))
        print("Fold Index", idx)
        print("테스트셋 버그 개수 :", len(bug_list[(idx - 1)*bug_train_len:idx*bug_train_len]))
        print("테스트셋 논 버그 개수 : ", len(non_bug_list[(idx - 1)*non_bug_train_len:idx*non_bug_train_len]))
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

    def _get_non_bug_copied_list(self, non_bug_set, set_num, non_bug_answer_set):
        new_bug_set = list()
        new_bug_answer_set = list()
        iter_num = int(set_num / len(non_bug_set))
        for i in range(0, iter_num):
            new_bug_set += non_bug_set
            new_bug_answer_set += non_bug_answer_set
        remain_num = set_num - len(new_bug_set)
        for i in range(0, remain_num):
            new_bug_set.append(non_bug_set[i])
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




