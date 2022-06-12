# -*- coding: utf-8 -*-
# @Author : LuoXianan
# @File : __init__.py.py
# @Project: Graduation design
# @CreateTime : 2022/5/6 18:09:00

from csv import reader   # 读取CSV文件
from random import randrange
from math import sqrt
from math import exp  # 算高斯
from math import pi

# 读取CSV  helper function  帮助主程序
def load_csv(filename):
    dataset = list()  # 生成一个空list
    with open(filename,'r',encoding='utf-8-sig') as file:
        csv_reader = reader(file)  # 调用reader读取file，生成读取器
        for row in csv_reader:  # 逐行读取
            if not row:  # 非空情况下
                continue
            dataset.append(row)  # 将每行读取的数据添加到空list
    return dataset

# # 测试 拿出CSV
# dataset = load_csv('data.csv')
# print(dataset)  # 打印出来的数据是字符串

# 数据类型转换
def convert_str_to_float(dataset,column):  # 把数据集和列放进去
    for row in dataset:  # 逐行转换
        row[column] = float(row[column].strip())  # 通过strip()对数据进行清洗，清洗数据前后的空格

# 类别字符串与int对应转换
def convert_str_to_int(dataset,column):  # 把数据集和需要转换的列放进去
    # class类别去重
    class_values = [row[column] for row in dataset]  # 逐行查找
    unique_value = set(class_values)  # 通过集合去重得到独特的值
    # 搞一个对应目录dictionary，然后进行一个look_up查询
    look_up =dict()   # look_up查询空dict
    # class和int对应，循环
    for i, value in enumerate(unique_value):  # 数字和class逐一，迭代每一个value
        look_up[value] = i  # 然后class和数字对应
    for row in dataset:  # 处理dataset里的数据
        row[column] = look_up[row[column]]  # 把每一个row的column替换为look_up具体对应的row里面的colum
    return look_up

# 数据训练集验证集切分，把数据k_fold交叉检验，把数据搞混，防止极端情况
def n_fold_cross_validation_split(dataset,n_folds):  # n_fold规定切多少次
    dataset_split = list()  # 先弄一个空list放着，然后在处理
    # 不要动原数据集，对原数据集进行copy，对这个copy的数据处理
    dataset_copy = list(dataset)  # copy原数据
    fold_size = int(len(dataset)/n_folds)  # 切分一个fold包含多少元素
    # 逐步把数据进行切分
    for i in range(n_folds):  # 切多少个循环多少次
        fold = list()
        while len(fold) < fold_size:  # 如果这个空list小于规定的fold_size
            index = randrange(len(dataset_copy))  # 通过index方式抽取
            fold.append(dataset_copy.pop(index))  # pop掉index的值，然后append到fold里
        dataset_split.append(fold)
    return dataset_split

# 准确性判断
def coculate_our_model_accuracy(actual,predicted):  # 真实数据和预测数据进行对比
    correct_count = 0  # 计数器
    for i in range(len(actual)):  # 遍历真实数据的个数
        if actual[i] == predicted[i]:  # 如果真实数据里面的第i个=预测数据的第i个
            correct_count += 1  # 计数器+1
    return correct_count/float(len(actual)) * 100.0  # 计算准确率

# 模型质量判断  #  algo指的是任何一个算法,具体有几个folds可以加进去
def whether_our_model_is_good_or_not(dataset,algo,n_folds, *args):
    folds = n_fold_cross_validation_split(dataset,n_folds)  # 复杂写库用之前写好的调用
    scores = list()  # 得分
    for fold in folds:  #
        train_set = list(folds)  # 训练集
        train_set.remove(fold)  # 把fold去掉
        train_set = sum(train_set,[])  # 更新train_set

        test_set = list()    # 测试集
        for row in fold:     # 逐行处理
            row_copy = list(row)   # copy原数据
            test_set.append(row_copy)  # copy的数据添加进test_set
            row_copy[-1] = None   # 为了预测的准确性，把结果去掉（防止模型抄答案）

        predicted = algo(train_set,test_set,*args)  # 预测数据，用某一种algo算法预测
        actual = [row[-1] for row in fold]  # 真实答案数据就是数据的最后一个，把最后一个数据填充进fold
        accracy = coculate_our_model_accuracy(actual,predicted)  # 计算准确率
        scores.append(accracy)  # 把计算的accracy 填充进scores
    return scores

# 按照种类进行数据切分
def split_our_data_by_class(dataset):
    splited = dict()   # 已经切分的数据放在这个dict()里  运算储存的工具
    for i in range(len(dataset)):  # 循环
        vector = dataset[i]     # vector = 第i个dataset
        class_value = vector[-1]   # class_value = 每条数据的最后一个
        if class_value not in splited :  # 如果这个class_value不在splited的情况下
            splited[class_value] = list()  # 把splited的里面的class_value放进一个空list里
        splited[class_value].append(vector)  # 然后把这个vector添加进splited
    return splited

# 平均值与标准差计算实现
# 可以把一个数据的基本信息去做相应的处理，可以直接调用一个打包好的函数，让这个函数帮我们搞出来
def mean(a_list_of_numbers):   # 平均值
    return sum(a_list_of_numbers)/float(len(a_list_of_numbers))  # 求和然后除总数得到平均值
# 计算标准差
# 每个class下每个特征的mean和stdev
# 这里的X为按照class分类后每个特征的值，逐个带入，就分别计算出
def stdev(a_list_of_numbers):
    the_mean_of_a_list_numbers = mean(a_list_of_numbers)  # 首先计算平均值
    # 计算方差  把a_list_of_numbers里的每一个元素都做一个运算，之后填充到list里再求和，再除以个数
    variance = sum([(x-the_mean_of_a_list_numbers)**2 for x in a_list_of_numbers])/float(
        len(a_list_of_numbers)-1)
    return sqrt(variance)   # 对vatiance开方

# 构建pandas数据 把一系列的dataset处理放在一个
def describe_our_data(dataset):
    description = [(mean(column),stdev(column),len(column))for column in zip(*dataset)]
    # 最后一列数据是class对我们没有用，删除掉，不需要最后一列的description
    del(description[-1])
    return description

# 按照class切分的数据总览
# 按照描述性数据的方式把class summarize做出来
def describe_our_data_by_class(dataset):  # 根据不同的class弄不同的description
    # 是在class判断的基础上再加description，可以直接调用之前的split_data
    data_split = split_our_data_by_class(dataset)   # 经过切分后的数据=之前的切分
    description = dict()  # 先弄一个空的出来，再做处理，再把空的进行填充
    for class_value,rows in data_split.items():  # 各种不同的class_value，以及不同的数据，从data_split里一条条取出来
        # 然后就可以对具体的diction里面的description的key和value做相应的更新
        # 对不同的class分类成0,1,2作为key弄进去 ，对于的value，把每行数据弄进去
        description[class_value] = describe_our_data(rows)
    return description

# 概率数据公式python计算，计算概率
# 唯一需要输入的变量就是x
def calculate_the_probability(x,mean,stdev):
    exponent = exp(-((x-mean)**2/(2*stdev**2 + 1e-8)))   # 除0错误，加上极小数
    result = (1/(sqrt(2*pi)*stdev + 1e-8)) * exponent    # 防止分母为0，加上极小数，对整体影响不大
    return result


# 复杂数据结构的探索方法 按照class分别计算概率
# 计算完概率必须要加上class，不同的类别它的probability是不一样的
# 可以把之前的东西全部调用然后加一个class,再写一个根据class进行计算的
# 因为description是根据class进行descrip ，同时每一行数据也需要
def calculate_the_probability_by_class(description,row):
    # 首先把所有的row求出来,逐行往里面走,以字典的形式放里面
    total_rows = sum([description[label][0][2] for label in description])
    probabilities = dict()    # 计算probabilities，先建一个空的，然后把class_value,class_description逐步放进去
    # 把class_value和class_description的值从descrip中通过item打开
    for class_value,class_description, in description.items():
        # 然后填充空的probabilities，key为class_value = 选中的某个具体的class/整体的row
        probabilities[class_value] = description[class_value][0][2] / float(total_rows)
        for i in range(len(class_description)):
            # 把mean,stedv,count解放出来
            mean,stdev,count = class_description[i]
            # 对probabilities进行相应的更新，把不同class_value里面的值进行一个计算
            probabilities[class_value] *= calculate_the_probability(row[i],mean,stdev)
    return probabilities

# 预测代码的实现
# 把每一行的数据，输入进去做预测，把每一行数据的总结性的东西放进去，然后逐行的处理
def predict(description,row):
    # 把probabilities进行一次计算
    probabilities = calculate_the_probability_by_class(description,row)  # probabilities = 调用之前计算概率的函数
    # 计算完之后得到最好的label和最好的概率，然后不断进行迭代更新，先放一个空的，然后更新到无法在更新就是最好的
    # best_label不知道，先放一个空的，best_prob从-1开始
    best_label,best_prob = None,-1
    # 逐步更新的代码，用for loop ，把不同的class_value和概率 通过items进行迭代
    for class_value,probability in probabilities.items():
        # 做一个判断，是否更新你原有的，如果原有的已经是best，就不用更新，如果没有就继续更新
        # 若果best_label是空或者目前的probabilit>best_prob
        if best_label is None or probability > best_prob:
            best_prob = probability   # 那么现在的probability就是best_prob
            best_label = class_value  # best_label就是那个时候对于的class_value
    return best_label  # 只返回了best_label,因为是朴素贝叶斯，只对最终的label感兴趣


# 底层代码实现后写一个总领的东西
# 朴素贝叶斯主体的实现 把train 和 test数据放进去进行计算
def naive_bayes(train,test):
    # 把train数据放进去，来训练模型，看看训练集的description
    description = describe_our_data_by_class(train)
    predictions = list()  # 生成一个专门装predicti的空list
    for row in test:  # 从训练集中取出每一row
        prediction = predict(description,row)  # 每次预测的prediction = 调用之前的predict
        predictions.append(prediction)  # 预测的集合列表 = 每次预测的集合
    return predictions


# 导入数据与数据预处理
dataset = load_csv('data.csv')  # dataset 读取到了数据集,读出来的是字符串
# 测试代码
# print(dataset)
# print('-'*40)
# 把字符串转换为float
for i in range(len(dataset[0])-1):  # -1 是因为最后一个数据是class,不用处理
    convert_str_to_float(dataset,i)  # 把数据集传进去，column就是第i个，做了一次转换
# 测试代码
# print(dataset)
# ----------------------------------------------------------------------------------------------
# 转换class 为数字
# 把dataset和column放进去，column是需要转换的列，在这里column就是最后一列的class，所以dataset[0]-1就是最后一列
convert_str_to_int(dataset,len(dataset[0])-1)
# 测试
# print('-'*40)
# print(dataset)
# -------------------------------------------------------------------------------------------------
fold = input('请输入切分次数：')
n_folds = int(fold)   # 切分的次数
data_split = n_fold_cross_validation_split(dataset,n_folds)   # n_fold规定切多少次
print('n_fold_cross_validation_split:\n')
print('-'*40)
# -------------------------------------------------------------------------------------
split = split_our_data_by_class(dataset)
print('split_our_data_by_class:\n')
print(split)
print('-'*40)
# ---------------------------------------------------------------------------------------
# describe our data
describetion = describe_our_data(dataset)
print(' describe our data:\n')
print(describetion)
print('-'*40)
# --------------------------------------------------------------------------------------------------
describe_our_data_by_class(dataset)
print('describe_our_data_by_class:\n')
describe = describe_our_data_by_class(dataset)
print(describe)
print('-'*40)

# ----------------------------------------------------------------------------------------

# 调用模型测试打分与调参

scores = whether_our_model_is_good_or_not(dataset,naive_bayes,n_folds)    # 给预测模型打分

#
#
print("This score of our model is :%s" %scores)
print('The accuracy of our model is :%s' %(sum(scores)/float(len(scores))))
#


































