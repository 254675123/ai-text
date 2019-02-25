import numpy as np
import pandas as pd
import re
from gensim import corpora, models, similarities
import gensim


def clean_email_text(text):
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text

if __name__ == "__main__":
    df = pd.read_csv("../../data/HillaryEmails.csv")
    # 原邮件数据中有很多Nan的值，直接扔了。
    df = df[['Id', 'ExtractedBodyText']].dropna()

    docs = df['ExtractedBodyText']
    docs = docs.apply(lambda s: clean_email_text(s))

    print(docs.head(1).values)
    doclist = docs.values

    stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
                'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
                'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
                'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
                'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
                'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
                'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
                'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
                'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
                'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
                'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
                'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

    texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print(corpus[13])
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
    the10 = lda.print_topic(10, topn=5)
    top20 = lda.print_topics(num_topics=20, num_words=5)
    topics = lda.get_topics()
    # 模型的保存/ 加载
    # lda.save('zhwiki_lda.model')
    # lda = models.ldamodel.LdaModel.load('zhwiki_lda.model')

    # 两个方法，我们可以把新鲜的文本 / 单词，分类成20个主题中的一个。
    # lda.get_document_topics(bow)
    # lda.get_term_topics(word_id)
    new_text = 'To all the little girls watching...never doubt that you are valuable and powerful & deserving of every chance & opportunity in the world.'
    new_clean_text = clean_email_text(new_text)
    new_word_list = [word for word in new_clean_text.lower().split() if word not in stoplist]
    new_corpus = dictionary.doc2bow(new_word_list)
    res = lda.get_document_topics(bow=new_corpus)
    print(res)
