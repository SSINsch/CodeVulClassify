# CodeBERT/GraphCodeBERT/translation/parser/utils.py 참조

from ._codetokenizer import CodeTokenizer
import logging

logger = logging.getLogger(__name__)


class CodeAbsTokenizer(CodeTokenizer):
    def __init__(self, lang='c'):
        super().__init__()
        self.variable_dict = {}

    def index_to_code_token(self, index, code):
        start_point = index[0]
        end_point = index[1]
        type_of_token = index[2]

        if start_point[0] == end_point[0]:
            s = code[start_point[0]][start_point[1]:end_point[1]]
        else:
            s = ""
            s += code[start_point[0]][start_point[1]:]
            for i in range(start_point[0] + 1, end_point[0]):
                s += code[i]
            s += code[end_point[0]][:end_point[1]]

        # identifier 타입인 경우(즉 변수명 등등) varN 으로 대체해서 보내기
        if 'identifier' in type_of_token:
            if s in self.variable_dict.keys():
                s = self.variable_dict[s]
            else:
                self.variable_dict[s] = 'var' + str(len(self.variable_dict))
                s = self.variable_dict[s]

        return s

    def tokenize(self, code):
        pass