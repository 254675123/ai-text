from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from file_util import binary_file

# 求得TF-IDF向量
def getTFIDF_Vector(bunch, stopWordList, outputPath, vocab=None):
    """
    给定分好词的bunch对象，停用词表，输出路径（用于保存结果）
    vocab用于区分train和test，当为train时，该参数不用传，默认为None
    当为test时，需要train的结果vocabulary作为参数
    :param bunch: 
    :param stopWordList: 
    :param outputPath: 
    :param vocab: 
    :return: 
    """
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, vector=[],
                       vocabulary=vocab)
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,vocabulary=vocab)

    # transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.vector = vectorizer.fit_transform(bunch.contents)
    if vocab is None:
        tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇

    binary_file.write_obj_by_pickle(outputPath, tfidfspace)

    return tfidfspace