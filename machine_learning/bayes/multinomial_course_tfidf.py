#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : homework3.py
# @Author: WangYe
# @Date  : 2018/4/22
# @Software: PyCharm
# 微博文字的性别识别
import jieba
import os
import pickle  # 持久化
from numpy import *
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法
from file_util import text_file,binary_file
from text_feature import tfidf_feature

def segmentText(inputPath):
    bunch = Bunch(target_name=[], label=[], contents=[])
    for line in text_file.readFile_line_onebyone(inputPath, coding='utf-8'):
        # 每行为：A2 西方列宁学研究
        # 是由空格分割的两部分，第一部分为code，第二部分为name
        section_list = line.split()
        if len(section_list) == 0:
            continue
        code = section_list[0]
        name = section_list[1]
        name_cut_list = jieba.cut(name)  # 默认方式分词，分词结果用空格隔开
        content = ' '.join(name_cut_list)
        if not bunch.target_name.__contains__(code):
            bunch.target_name.append(code)
        bunch.label.append(code)
        bunch.contents.append(content)
        print(content)
    print("read over")
    return bunch

def getTFIDFMat(bunch, stopWordList, outputPath):  # 求得TF-IDF向量
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, tdm=[],
                       vocabulary={})
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    # transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇
    binary_file.write_obj_by_pickle(outputPath, tfidfspace)


def getTestSpace(bunch, trainSpacePath, stopWordList, testSpacePath):
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, content=bunch.contents, label=bunch.label, tdm=[],
                      vocabulary={})
    # 导入训练集的词袋
    trainbunch = binary_file.read_obj_by_pickle(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    # transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    # 持久化
    binary_file.write_obj_by_pickle(testSpacePath, testSpace)


def bayesAlgorithm(trainPath, testPath):
    trainSet = binary_file.read_obj_by_pickle(trainPath)
    testSet = binary_file.read_obj_by_pickle(testPath)
    clf = MultinomialNB(alpha=0.001).fit(trainSet.vector, trainSet.label)
    # alpha:0.001 alpha 越小，迭代次数越多，精度越高
    # print(shape(trainSet.tdm))  #输出单词矩阵的类型
    # print(shape(testSet.tdm))
    predicted = clf.predict(testSet.vector)
    total = len(predicted)
    rate = 0
    for flabel,  expct_cate in zip(testSet.label,  predicted):
        if flabel != expct_cate:
            rate += 1
            print("实际类别：", flabel, "-->预测类别：", expct_cate)
    print("erroe rate:", float(rate) * 100 / float(total), "%")


root_path = './../../data/text-corpus-catalog-course'
# 分词，第一个是分词输入，第二个参数是结果保存的路径
bunch = segmentText(root_path+"/catalog_course-train.txt")
# 获取停用词
stopWordList = text_file.readFile_lines_once(root_path+"/stopword.txt", coding='utf-8')
#getTFIDFMat(bunch, stopWordList, root_path+"/tfidfspace-train.dat")  # 输入词向量，输出特征空间
train_bunch = tfidf_feature.getTFIDF_Vector(bunch, stopWordList, root_path+"/tfidfspace-train.dat")
#print(shape(train_bunch.vector))
# 训练集
bunch = segmentText(root_path+"/catalog_course-test.txt")  # 分词
#getTestSpace(bunch, root_path+"/tfidfspace-train.dat", stopWordList, root_path+"/tfidfspace-test.dat")
test_bunch = tfidf_feature.getTFIDF_Vector(bunch, stopWordList, root_path+"/tfidfspace-test.dat", train_bunch.vocabulary)
print(shape(test_bunch.vector))
bayesAlgorithm(root_path+"/tfidfspace-train.dat", root_path+"/tfidfspace-test.dat")

