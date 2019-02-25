# -*- coding: utf-8 -*-
# @Author: zhangchaolin
# @Date  : 2018/12/01
# @Software: pycharm
# 文本文件的读取和写入
# Python 将文本文件的内容读入可以操作的字符串变量非常容易。
# 文件对象提供了三个“读”方法： .read()、.readline() 和 .readlines()。
# 每种方法可以接受一个变量以限制每次读取的数据量，但它们通常不使用变量。
# .read() 每次读取整个文件，它通常用于将文件内容放到一个字符串变量中。
# 然而 .read() 生成文件内容最直接的字符串表示，但对于连续的面向行的处理，它却是不必要的，
# 并且如果文件大于可用内存，则不可能实现这种处理。
# .readline() 和 .readlines() 非常相似。
# .readline() 和 .readlines() 之间的差异是后者一次读取整个文件，象 .read() 一样。
# .readlines() 自动将文件内容分析成一个行的列表，该列表可以由 Python 的 for … in … 结构进行处理。
# 另一方面，.readline() 每次只读取一行，通常比 .readlines() 慢得多。
# 仅当没有足够内存可以一次读取整个文件时，才应该使用 .readline()。


def readFile_lines_once(path, coding=None):
    """
    一次性读取所有行，文件较小时使用，返回文件内容的行列表
    :param path: 
    :param coding: 
    :return: 
    """
    lines = []
    with open(path, 'r',encoding=coding ,errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        line_list = file.readlines()
        for line in line_list:
            lines.append(line)
    return lines


def readFile_line_onebyone(path, coding=None):
    """
    一次性读取一行，文件较大时使用，返回文件内容的一行
    文档中编码有问题时，用errors过滤错误
    :param path: 
    :param coding: 
    :return: 
    """
    with open(path, 'r',encoding=coding ,errors='ignore') as file:
        line = file.readline()
        yield line

        while line:
            line = file.readline()
            yield line


def saveFile_content_once(path, content):
    """
    file.write(str)的参数是一个字符串，就是你要写入文件的内容. write输出后不换行
    file.writeline(str) 输出后换行
    file.writelines(sequence)的参数是序列，比如列表，它会迭代帮你写入文件。
    :param path: 
    :param content: 
    :return: 
    """
    with open(path, 'a', errors='ignore') as file:
        file.write(content)

def saveFile_lines_once(path, lines):
    """
    file.write(str)的参数是一个字符串，就是你要写入文件的内容. write输出后不换行
    file.writeline(str) 输出后换行
    file.writelines(sequence)的参数是序列，比如列表，它会迭代帮你写入文件。
    :param path: 
    :param lines: 
    :return: 
    """
    with open(path, 'a', errors='ignore') as file:
        file.writelines(lines)