import numpy as np
import os
import cv2


class batchGenerator:
    def __init__(self, basePath='data/processed/train_224/', batchSize=256):
        self.basePath = basePath
        # 读取全部文件名
        self.fileList = os.listdir(self.basePath)
        # 打乱文件名顺序
        for i in range(10):
            np.random.shuffle(self.fileList)
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
        if endIndex >= self.num_files:
            endIndex = None

        # 当前batch再次打乱顺序
        curSampleList = [fileName for fileName in self.fileList[self.curIndex:endIndex]]
        np.random.shuffle(curSampleList)

        for fileName in curSampleList:
            # 读取当前图片
            file = os.path.join(self.basePath, fileName)
            image = cv2.imread(file)
            # 确定当前图片标签
            cur_type = fileName.split('(')[0].strip()
            try:
                curLabel = np.zeros(self.num_labels)
                curLabel[self.labels.index(cur_type)] = 1
            except:
                print('file name error')
                print(fileName)
                exit()

            # 添加值到待返回的列表
            curBatchX.append(list(image))
            curBatchY.append(curLabel)

        # 改变curIndex的值
        self.curIndex = endIndex
        if endIndex is None:
            np.random.shuffle(self.fileList)
            self.curIndex = 0

        return np.array(curBatchX), np.array(curBatchY)

