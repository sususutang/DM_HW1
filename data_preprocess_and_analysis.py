import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency

dataset1_path = "./github_dataset.csv"
dataset2_path = "./movies_dataset.csv"
Movies_Nominal_Attribute = ['appropriate_for','director', 'id', 'industry', 'language', 'posted_date', 'release_date', 'run_time', 'storyline', 'title', 'writer']
Movies_Number_Attribute = ['IMDb-rating', 'downloads', 'views']
Github_Nominal_Attribute = ['repositories', 'language']
Github_Number_Attribute = ['stars_count', 'forks_count', 'issues_count', 'pull_requests', 'contributors']

def Movies_Data_summarization_and_visualization(df):
    # 遍历每一列数据
    num_attr = []
    nom_attr = []
    # 预处理数据
    for e in range(1,3):
        #print(e)
        for i, s in zip(range(len(df[Movies_Number_Attribute[e]])), df[Movies_Number_Attribute[e]]):
            if type(s) == str and s.__contains__(','):
                df[Movies_Number_Attribute[e]][i] = s.replace(',', '')
        df[Movies_Number_Attribute[e]] = df[Movies_Number_Attribute[e]].astype(float)
    numerical_datas = df[Movies_Number_Attribute]

    for col_name in df.columns:
        if col_name == "Unnamed: 0":
            continue
        if col_name in Movies_Number_Attribute:
            num_attr.append(col_name)
            # 统计五数概括和缺失值个数
            attribute = df[col_name]
            five_num = attribute.describe()[['min', '25%', '50%', '75%', 'max']]
            missing_values = attribute.isnull().sum()
            freq_count = attribute.value_counts()
            print("  五数概括：\n", five_num)
            print("  缺失值个数：", missing_values)
            Q1 = attribute.quantile(0.25)
            Q3 = attribute.quantile(0.75)
            outliner = Q3 + (Q3 - Q1) * 1.5
            print(f"大于{outliner}的项被识别为离群点")

            # 绘制盒图
            plt.boxplot(attribute.dropna().values)
            plt.title(col_name + "  boxplot")
            #plt.show()
            plt.savefig("./img/Movies/"+col_name+"_boxplot.png")
        elif col_name in Movies_Nominal_Attribute:
            nom_attr.append(col_name)
            # 统计每个可能取值的频数
            attribute = df[col_name]
            freq_count = attribute.value_counts()
            print("  频数统计：\n", freq_count)

        # 绘制直方图，由于标称属性的直方图不直观，故对于标称属性只针对频数最大的十个绘制直方图            
        if col_name in Movies_Nominal_Attribute:
            freq_count[:10].plot(kind="bar", figsize=(10,4))
            plt.xticks(rotation=90)
            plt.title(col_name + "  histogram")
            #plt.show()
            plt.savefig("./img/Movies/"+col_name+"_histogram.png")
        else:
            plt.hist(attribute.dropna().values, bins=len(freq_count))
            plt.xticks(rotation=90)
            plt.title(col_name + "  histogram")
            #plt.show()
            plt.savefig("./img/Movies/"+col_name+"_histogram.png")

    print("Movies数值属性名称列表:", num_attr)
    print("Movies标称属性名称列表:", nom_attr)


def Movies_Data_process(df):
    # 策略1：将缺失部分剔除
    df1 = df.dropna()

    # 策略2：用最高频率值来填补缺失值
    df2 = df.fillna(df.mode().iloc[0])

    # 策略3：通过属性的相关关系来填补缺失值
    # 计算属性相关性
    corr_matrix = df.corr()
    print(corr_matrix)
    df_copy = df
    # 先用众数补全'views'
    mode_Low_Confidence_Limit = df_copy['views'].mode()[0]
    df_copy['views'].fillna(mode_Low_Confidence_Limit, inplace=True)
    # 选取一些与目标特征有一定相关性的特征作为自变量
    corr_attributes = ['views']
    df3 = df_copy
    # 构建预测模型
    lr = LinearRegression()
    train_data = df3[df3['downloads'].notnull()]
    test_data = df3[df3['downloads'].isnull()]
    X_train = train_data[corr_attributes]
    y_train = train_data['downloads']
    X_test = test_data[corr_attributes]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    # 填充预测值
    df3.loc[df3['downloads'].isnull(), 'downloads'] = y_pred

    # 策略4：通过数据对象之间的相似性来填补缺失值
    df4 = df.copy()
    imp = KNNImputer(n_neighbors=5, weights='uniform')
    df4[Movies_Number_Attribute] = imp.fit_transform(df4[Movies_Number_Attribute])

    # 输出处理后的数据集
    print("策略1:将缺失部分剔除")
    print(df1.isnull().sum())

    print("\n策略2:用最高频率值来填补缺失值")
    print(df2.isnull().sum())

    print("\n策略3:通过属性的相关关系来填补缺失值")
    print(df3.isnull().sum())

    print("\n策略4:通过数据对象之间的相似性来填补缺失值")
    print(df4.isnull().sum())

    # 导出新数据集
    df1.to_csv('Movies_data_1.csv', index=False)
    df2.to_csv('Movies_data_2.csv', index=False)
    df3.to_csv('Movies_data_3.csv', index=False)
    df4.to_csv('Movies_data_4.csv', index=False)

def Github_Data_summarization_and_visualization(df):
    # 遍历每一列数据
    num_attr = []
    nom_attr = []
    for col_name in df.columns:
        if col_name in Github_Number_Attribute:
            num_attr.append(col_name)
            # 统计五数概括和缺失值个数
            attribute = df[col_name]
            five_num = attribute.describe()[['min', '25%', '50%', '75%', 'max']]
            missing_values = attribute.isnull().sum()
            freq_count = attribute.value_counts()
            print("  五数概括：\n", five_num)
            print("  缺失值个数：", missing_values)
            Q1 = attribute.quantile(0.25)
            Q3 = attribute.quantile(0.75)
            outliner = Q3 + (Q3 - Q1) * 1.5
            print(f"大于{outliner}的项被识别为离群点")

            # 绘制盒图
            plt.boxplot(attribute.dropna().values)
            plt.title(col_name + "  boxplot")
            #plt.show()
            plt.savefig("./img/Github/"+col_name+"_boxplot.png")
        elif col_name in Github_Nominal_Attribute:
            nom_attr.append(col_name)
            # 统计每个可能取值的频数
            attribute = df[col_name]
            freq_count = attribute.value_counts()
            print("  频数统计：\n", freq_count)

        # 绘制直方图
        plt.hist(attribute.dropna().values, bins=len(freq_count))
        plt.xticks(rotation=90)
        plt.title(col_name + "  histogram")
        #plt.show()
        plt.savefig("./img/Github/"+col_name+"_histogram.png")

    print("Github数值属性名称列表:", num_attr)
    print("Github标称属性名称列表:", nom_attr)


def Github_Data_process(df):
    # 在language列存在缺失值
    # 策略1：将缺失部分剔除
    df1 = df.dropna()

    # 策略2：用最高频率值来填补缺失值
    df2 = df.fillna(df.mode().iloc[0])

    # 策略3：通过属性的相关关系来填补缺失值
    corr_matrix = df.corr()
    print(corr_matrix)
    # language 与其他属性不存在相关关系

    # 策略4：通过数据对象之间的相似性来填补缺失值
    df4 = df.copy()
    imp = KNNImputer(n_neighbors=5)
    #print(imp.fit_transform(df4))
    # language 是object，标称属性，字符串不适合使用数据对象的相似性来填补
    

    # 输出处理后的数据集
    print("策略1:将缺失部分剔除")
    print(df1.isnull().sum())

    print("\n策略2:用最高频率值来填补缺失值")
    print(df2.isnull().sum())

    # 导出新数据集
    df1.to_csv('Github_data_1.csv', index=False)
    df2.to_csv('Github_data_2.csv', index=False)


if __name__ == "__main__":
    df1 = pd.read_csv(dataset1_path)
    df2 = pd.read_csv(dataset2_path,encoding='ISO-8859-1')
    # print(df1.isnull().sum())
    # print(df2.isnull().sum())
    Github_Data_summarization_and_visualization(df=df1)
    Github_Data_process(df=df1)
    Movies_Data_summarization_and_visualization(df=df2)
    Movies_Data_process(df=df2)

    
