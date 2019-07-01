import numpy as np
import os
import cv2


class batchGenerator:
    def __init__(self, basePath='..\\step1\\processed\\train_224\\', batchSize=256):
        self.basePath = basePath
        # 读取全部文件名
        self.fileList = os.listdir(self.basePath)
        # 打乱文件名顺序

        # 记录总样本数
        self.num_files = len(self.fileList)
        # 记录现在样本索引的游标，每次读取bacth后，游标像后移动
        # 一个epoch后,即文件读完时,游标回到 0
        self.curIndex = 0
        # 该生成器每次返回的样本数量(最后一次返回的数量为 总数%batchSize )
        self.batchSize = batchSize
        self.labels = ['bus','family sedan','fire engine','racing car']
        self.num_labels = len(self.labels)

    def getBatch(self):

        # 记录当前batch的图片值和对应的标签
        curBatchX = []
        curBatchY = []

        endIndex = self.curIndex + self.batchSize
        # 如果endIndex超过了list的长度，应该设置为None

        for fileName in self.fileList[self.curIndex:endIndex]:
            # 读取当前图片
            file = os.path.join(self.basePath, fileName)
            image = cv2.imread(file)
            # 确定当前图片标签
            cur_type = fileName.split('(')[0].strip()
            # 需要取消注释的地方请取消注释
            # try:
                # 构造标签  例如是bus 则curLabel = [1,0,0,0]

            # except:
            #     if fileName != '.gitignore':
            #         print('file name error')
            #         print(fileName)
            #         exit()
            #     else:
            #         continue

            # 将（224，224，3）转化为（3，224，224）
            # 新的数组，三维分别是三个通道
            # imageNew = np.transpose(image, (2, 0, 1))

            # 添加值到待返回的列表
            # curBatchX.append(list(image))
            # curBatchY.append(curLabel)

        # 改变curIndex的值


        return np.array(curBatchX), np.array(curBatchY)
