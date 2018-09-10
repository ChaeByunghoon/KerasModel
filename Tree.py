

class Node:

    def __init__(self, file_name, node_list, is_buggy=0):
        self.file_name = file_name
        self.is_buggy = is_buggy
        self.node_list = node_list

    def set_file_name(self, file_name, node_list):
        self.file_name = file_name
        self.node_list = node_list

    def set_is_buggy(self, is_buggy):
        self.is_buggy = is_buggy
