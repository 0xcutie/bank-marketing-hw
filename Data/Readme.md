原始数据 bank-addtional.csv
10折训练测试数据：
1.删除不相关列：['default','pdays']
2.处理分类变量 缺失值众数填充（Mode Impute）+ Label Encoding + One-Hot Encoding ：["job", "marital", "education", "housing", "loan", "contact", "poutcome"]
3.处理数值特征 缺失值填充（均值填充）+ 标准化 ：["age", "campaign", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
