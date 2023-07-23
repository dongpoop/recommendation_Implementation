# recommendation_Implementation

这是一个基于协同过滤算法思想的电影推荐系统实现。

数据：MovieLens数据集（只使用了ratings表）。

环境：python（3.10），pycharm，MySql 8.0+

## 如何使用：

1，只需要下载这个仓库中的collaborative_filtering_for_movies.py文件，导入依赖。

2.从MovieLens下载数据导入到Mysql中。

3.直接运行。

## 说明：

1.similarity是相似数据，当你运行collaborative_filtering_for_movies.py的时候会自动产生，不需要使用，之所以上传到仓库只是为了便于展示项目。

2.test.py是测试代码，用来测试算法的准确率（准确来说，是精度）。

## 算法思想：

### collaborative_filtering_for_movies.py

1.使用从数据库得到的数据在内存中构建临时的数据存储结构（代码中称为评分矩阵，这一步还要对未评分数据进行填充），如下：

                  |      1     2     3     4     5    6   movie_id
         _________|_____________________________________________________________________________________________________
               1  |     3.2   3.1  5.0
         _________|_____________________________________________________________________________________________________
               2  |           2.1               4.0
         _________|_____________________________________________________________________________________________________
               3  |
         _________|_____________________________________________________________________________________________________
          user_id |
                  |
                  
2.计算上图每两列的相似度（有多种方法，我只写了调整的余弦相似度，但是我预留了其他的算法接口，你可以自己定义相似度计算方法），存入**当前目录**下的similarity.data
中（是个csv文件，可以用excel打开）。

3.为特定的用户推荐电影。

### test.py

进行n(你自己输入)次测试。每次测试取全集的一定比例（同样自己输入这一参数，取出的数据作为训练集，调用collaborative_filtering_for_movies.py中的推荐函数），
再到测试集中计算每一个用户的精度后，取所有训练集用户的精度的均值，作为准确率。
