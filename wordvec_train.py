import sys

import flag
import WordVecModeler_new
import FileController

m = WordVecModeler_new.WordVecModeler(flag.node_dim)

manager = FileController.FileManager("../Dataset/final_data")

convertList = list()
option = int(sys.argv[1])
output_name = str(sys.argv[2])
isCBOW = int(sys.argv[3])
if sys.argv[1] is None or output_name is None:
    print("인자를 입력해라")

for data in manager.except_infrequent_word(manager.all_node_list):
    # 여기서 직렬화 옵션 조정
    convertList.append(utils_new.convert_node_to_vector_list(data, option=option))

print("finish read files.")

model = m.train_word_vec(convertList, output_name, epoch=200, sg=isCBOW)

print("model train finished")