# -*- coding: utf-8 -*-
# @Author: zhangchaolin
# @Date  : 2018/12/01
# @Software: pycharm
# 二进制文件的读取和写入

import pickle

def read_obj_by_pickle(path):
    """
    使用pickle从二进制文件反序列化为对象
    :param path: 
    :return: 
    """
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def write_obj_by_pickle(path, obj_data):
    """
    序列化对象为二进制文件
    :param path: 
    :param obj_data: 
    :return: 
    """
    with open(path, 'wb') as file:
        pickle.dump(obj_data, file)
