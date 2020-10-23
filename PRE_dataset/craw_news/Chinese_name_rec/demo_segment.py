# -*- coding: utf-8 -*-

# 分词demo
# Author: Alex
# Created Time: 2017年04月17日 星期一 16时51分05秒

from jpype import *

hanLPLibPath = '/var/hanlp/hanlp-1.3.2-release/'
javaClassPath = hanLPLibPath+'hanlp-1.3.2.jar'+':'+hanLPLibPath

startJVM(getDefaultJVMPath(), '-Djava.class.path='+javaClassPath, '-Xms1g', '-Xmx1g')
HanLP = JClass('com.hankcs.hanlp.HanLP')


def InfoTitle(title):
    print("")
    print("*"*10, title, "*"*10)

# text = '中国科学院计算技术研究所的宗成庆教授正在教授自然语言处理课程'
# text = '数据雷达是一个外部数据分析平台，帮助商家掌握行业趋势，了解竞争对手的状况等。'
text = '陆星儿简介陆星儿，祖籍江苏海门，1949年11月生于上海。1968年前在上海读书；1978年考入中央戏剧学院戏剧文学系；1982年始任中国儿童艺术剧院编剧；1983年加入中国作家协会，生前为上海作家协会专业作家。出版长篇小说五部、散文集7部、中短篇小说集10部、影视剧多部。陆星儿最让人敬佩的是毅力。本报讯（记者甘丹）9月4日晚8时30分，著名女作家陆星儿因癌症医治无效，在上海去世，享年55岁。记者了解到，陆星儿的病情一直就非常严重。在经过第二次手术以后，癌症细胞甚至已经扩散到了全身很多器官。然而她并不太愿意大家过多地提到她的病情，她总是不断地鼓励自己，希望自己能坚强地活下去。陆星儿在生病期间创作了长篇小说《痛》、以及长篇散文《用力呼吸———陆星儿生命日记》，这对于一个身患癌症的人来说是件极其不容易的事情。因《寻枪》而出名的导演陆川是陆星儿的侄子，他非常伤心，并告诉记者他父亲、陆星儿的哥哥著名作家陆天明更为悲伤，已经飞赴上海。作家陈村在接受记者采访时说：“她在生命的最后，有儿子和许多朋友陪在她的身边，也许还算一种安慰吧。”亲友追忆]陆川：有这样的姑姑我很荣幸在明白记者采访目的后，陆川沉默了好几秒。然后他说：“我真的非常伤心。”昨天晚上做完了《可可西里》的拷贝，试放到电影结尾时，接到了父亲（陆天明）的电话，他告诉我星儿姑姑去世了，真的非常伤心。我本来应该去上海的，但现在正赶着做《可可西里》的拷贝，离不开。我父母今天早上已经飞回上海，我随后再过去。姑姑对我很好，我也很喜欢他。一直以来我都以有一个这样的姑姑而感到荣幸，因为她和我的父亲都是从事文学的，我觉得很了不起。她的病痛让她很痛苦，但是姑姑却很坚强。现在离开对她来说也许是一种解脱，我相信在另外一个国度或者世界，姑姑一定得到了永恒！陈村：她的一生都是坎坷的陆星儿平时是一个很低调的人，从不炒作。在工作上，她却特别的刻苦、勤劳。回想她的一生，是那么的不顺利。她从年轻时下乡到北大荒，到后来离婚一个人带着儿子生活，再到生病都是坎坎坷坷的，我们这些朋友在心里都很为她难过。她的儿子叫陈厦，为什么呢？那是因为，当年她怀着孩子的时候，大着肚子坐在她丈夫的自行车上到处奔波，就是为了找房子。他们来来回回搬了许多次家，很长一段时间都是处于这种颠簸的状态，所以她就给孩子取名叫陈厦，希望自己的下一代不会再有这样的经历。她平时很节俭，总是把经历放在了工作上、友情上。上个月我最后一次见到她的时候，她又比以前瘦了好多。我们其实也就料想到了结局，但是这结局又不免让我们难过。陆星儿简介陆星儿，祖籍江苏海门，1949年11月生于上海。1968年前在上海读书；1978年考入中央戏剧学院戏剧文学系；1982年始任中国儿童艺术剧院编剧；1983年加入中国作家协会，生前为上海作家协会专业作家。出版长篇小说五部、散文集7部、中短篇小说集10部、影视剧多部。'

# 标准分词
# 标准分词是最常用的分词器，基于HMM-Viterbi实现，开启了中国人名识别和音译人名识别
# # HanLP.segment 其实是对 StandardTokenizer.segment 的包装。
# # HanLP中有一系列“开箱即用”的静态分词器，以 Tokenizer 结尾
# InfoTitle("标准分词")
# termList = HanLP.segment(text)
# print(termList)
# for i in range(len(termList)):
#     term = termList[i]
#     print(i, term)
#
# # NLP分词
# # NLPTokenizer 会执行全部命名实体识别和词性标注。
# # 速度比标准分词慢，并且有误识别的情况。
# InfoTitle("NLP分词")
# tokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
# print(tokenizer.segment(text))
#
# # 索引分词
# # IndexTokenizer 是面向搜索引擎的分词器，能够对长词全切分，另外通过
# # term.offset 可以获取单词在文本中的偏移量。
# InfoTitle("索引分词")
# tokenizer = JClass('com.hankcs.hanlp.tokenizer.IndexTokenizer')
# termList = tokenizer.segment(text)
# for i in range(len(termList)):
#     term = termList[i]
#     print(term, " \t", term.offset, ":", term.offset + len(term.word))

# 通过HanLP.newSegment
# 通过此工厂方法得到的是当前版本速度和效果最平衡的分词器
# 推荐用户始终通过工具类HanLP调用，这么做的好处是，将来HanLP升级后，用户无需修改调用代码。
InfoTitle("通过HanLP.newSegment")
segment = HanLP.newSegment().enableNameRecognize(True)\
    .enableTranslatedNameRecognize(True)\
    .enablePlaceRecognize(True)\
    .enableOrganizationRecognize(True)
# 注意
# 报错：RuntimeError: Multiple overloads possible.
termList = segment.seg(JString(text))
print(termList)

shutdownJVM()
