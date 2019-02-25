# -*- coding:utf-8 -*-

import os
import sys
import gensim
from gensim.models import Doc2Vec
import jieba

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class TextVector:
    """
    文本向量，计算2个文本句子之间的相似度
    """

    def __init__(self):
        """
        initialize local variables.
        """
        # 分类目录数据
        self.catalog_code_dict = {}

        self.stopwords = [u'学', u'类', u'中国', u'国际', u'国外', u'西方']
        # 获取第二层的分类
        self.snd_level_catalog = []

        self.model = None

        self.index_catalog = {}

    def generate_train_file(self):
        pass


    def train(self):
        # 加载数据
        documents = []
        # 使用count当做每个句子的“标签”，标签和每个句子是一一对应的
        count = 0
        for words_tuple in self.snd_level_catalog:

            words = words_tuple[3]
            self.index_catalog[count] = words_tuple[1]
            # 这里documents里的每个元素是二元组，具体可以查看函数文档
            documents.append(gensim.models.doc2vec.TaggedDocument(words, [str(count)]))
            count += 1
            if count % 10000 == 0:
                print('{} has loaded...'.format(count))

        # 模型训练
        self.model = Doc2Vec(dm=1, vector_size=200, window=8, min_count=1, workers=4, epochs=2000)
        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        # 保存模型
        model_file = u'D:/pythonproject/open-neo4j-service/data/course-base/本科专业目录-catalog.d2v.model'
        self.model.save(model_file)

    def test_doc2vec(self):
        # 加载模型
        #model = Doc2Vec.load('models/ko_d2v.model')
        model = self.model
        # 与标签‘0’最相似的
        print(model.docvecs.most_similar('0'))
        # 进行相关性比较
        print(model.docvecs.similarity('0', '1'))
        # 输出标签为‘10’句子的向量
        print(model.docvecs['10'])
        # 也可以推断一个句向量(未出现在语料中)
        #words = u"여기 나오는 팀 다 가슴"
        course_name = u'比较教育学'
        words = jieba.cut(course_name)
        vector = model.infer_vector(words)
        sims = model.docvecs.most_similar([vector], topn=len(model.docvecs))

        for sim in sims:
            name = self.index_catalog.get(int(sim[0]))
            print('{}, {}'.format(name, sim[1]))

        # 也可以输出词向量
        #print(model[u'가슴'])



if __name__ == "__main__":
    tv = TextVector()
    tv.generate_train_file()
    tv.train()
    tv.test_doc2vec()
