# coding: utf-8
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
'''
iris数据一共有150组，前100组作为训练样本，后50组作为测试样本
'''


def predict_train(x_train, y_train):
    '''
    使用信息熵作为划分标准，对决策树进行训练
    参考链接： http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    '''
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    # print(clf)
    clf.fit(x_train, y_train)
    ''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    print 'feature_importances_: %s' % clf.feature_importances_

    '''测试结果的打印'''
    y_pre = clf.predict(x_train)
    # print(x_train)
    print "结果做差"
    cha = y_pre - y_train
    print(cha)
    print(np.mean(y_pre == y_train))
    return y_pre, clf



def show_pdf(clf):
    '''
    可视化输出
    把决策树结构写入文件: http://sklearn.lzjqsdd.com/modules/tree.html

    Mac报错：pydotplus.graphviz.InvocationException: GraphViz's executables not found
    解决方案：sudo brew install graphviz
    参考写入： http://www.jianshu.com/p/59b510bafb4d
    '''
    # with open("testResult/tree.dot", 'w') as f:
    #     from sklearn.externals.six import StringIO
    #     tree.export_graphviz(clf, out_file=f)

    import pydotplus
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("/home/wendong/PycharmProjects/sklearn/output/tree1.pdf")



if __name__ == '__main__':
    iris = load_iris()

    # 前140组作为训练的数据
    train_data = iris.data[:140]
    train_lable = iris.target[:140]
    print(train_data.shape[0])
    print(iris.target)

    # 采用决策树训练模型
    ypre, clf = predict_train(train_data, train_lable)
    #将决策树打印出来
    show_pdf(clf)
    # 后10组作为测试数据
    test_data = iris.data[140:]
    test_lable = iris.target[140:]
    test_pre = clf.predict(test_data)
    print("测试结果做差")
    cha = test_pre - test_lable
    print(cha)