# tinyml
利用numpy实现的一些周志华《机器学习》（西瓜书）一书及 斯坦福cs229课程中的算法，宜配合西瓜书和cs229课件食用。并选择性实现了一些经典算法的简易版本，
如 按照陈天奇的slides实现的XGBRegressor。
## 已经实现的算法
- **线性模型**  
-[x] [LinearRegression](/tinyml/linear_model/LinearRegression.py)    [线性回归闭式解推导](notes/linear_model/linear_reg_closed_form.pdf)  
-[x] [LogisticRegression](/tinyml/linear_model/LogisticRegression.py)  [逻辑回归相关推导](/notes/linear_model/logistic_regression.pdf)  
-[x] [SGDRegressor](/tinyml/linear_model/SGDRegressor.py)  
-[x] [LocallyWeightedLinearRegression](/tinyml/linear_model/LocallyWeightedLinearRegression.py)    
- **判别分析**  
-[x] [LDA](/tinyml/discriminant_analysis/LDA.py)  
-[x] [GDA](/tinyml/discriminant_analysis/GDA.py)    
- **决策回归树**   
-[x] [DecisionTreeClassifier](/tinyml/tree/DecisionTreeClassifier.py)  
-[x] [DecisionTreeRegressor](/tinyml/tree/DecisionTreeRegressor.py)    
- **支持向量机**  
-[x] [SVC](/tinyml/svm/SVC.py)  
- **贝叶斯**  
-[x] [NaiveBayesClassifier](/tinyml/bayes/NaiveBayesClassifier.py)  
- **聚类算法**
-[x] [KMeans](/tinyml/cluster/KMeans.py)  
-[x] [LVQ](/tinyml/cluster/LVQ.py)  
-[x] [GaussianMixture](/tinyml/cluster/GaussianMixture.py)  
-[x] [DBSCAN](/tinyml/cluster/DBSCAN.py)  
-[x] [AGNES](/tinyml/cluster/AGNES.py)    
- **降维算法**  
-[x] [MDS](/tinyml/dimension_reduction/MDS.py)  
-[x] [PCA](/tinyml/dimension_reduction/PCA.py)  
-[x] [KernelPCA](/tinyml/dimension_reduction/KernelPCA.py)  
-[x] [LLE](/tinyml/dimension_reduction/LLE.py)  
-[x] [Isomap](/tinyml/dimension_reduction/Isomap.py)    
- **集成学习**  
-[x] [AdaBoostClassifier](/tinyml/ensemble/AdaBoostClassifier.py)
-[x] [GradientBoostingRegressor](/tinyml/ensemble/GradientBoostingRegressor.py)  
-[x] [RandomForestRegressor](/tinyml/ensemble/RandomForestRegressor.py)  
-[x] [XGBRegressor](/tinyml/ensemble/XGBRegressor.py)    
- **特征选择**  
-[x] [ReliefFeatureSelection](/tinyml/feature_selection/ReliefFeatureSelection.py)  
## 和sklearn实现的比较
- 回归算法结果 [代码](/tinyml/compare/compare_regresssor.py)
<table>
    <tr>
         <td rowspan="2">Algorithm vs. RMSE</td>
        <td colspan="2">sklearn-boston</td>
    </tr>
    <tr>
        <td></td>
        <td>tinyml</td>
        <td>sklearn</td>
    </tr>
    <tr>
        <td>LinearRegression</td>
        <td>27.196</td>
        <td>27.196</td>
    </tr>
    <tr>
        <td>SGDRegressor</td>
        <td>27.246</td>
        <td>27.231</td>
    </tr>
    <tr>
        <td>DecisionTreeRegressor</td>
        <td>21.887</td>
        <td>21.761</td>
    </tr>
    <tr>
        <td>RandomForestRegressor</td>
        <td>21.142</td>
        <td>21.142</td>
    </tr>
    <tr>
        <td>GradientBoostRegressor</td>
        <td>16.778</td>
        <td>16.106</td>
    </tr>
    <tr>
        <td>XGBRegressor</td>
        <td>20.149</td>
        <td>15.7</td>
    </tr>
</table>

- 分类算法结果 [代码](/tinyml/compare/compare_classification.py)
<table>
   <tr>
      <td>Algorithm vs. accuracy</td>
      <td>sklearn-breast_cancer</td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td>   tinyml</td>
      <td>sklearn</td>
   </tr>
   <tr>
      <td>LogisticRegression</td>
      <td>92.98%</td>
      <td>92.98%</td>
   </tr>
   <tr>
      <td>LDA</td>
      <td>94.15%</td>
      <td>92.40%</td>
   </tr>
   <tr>
      <td>GDA</td>
      <td>92.40%</td>
      <td>93.57%</td>
   </tr>
   <tr>
      <td>SVC</td>
      <td>86.55%</td>
      <td>92.98%</td>
   </tr>
   <tr>
      <td>AdaboostClassifier</td>
      <td>92.40%</td>
      <td>92.40%</td>
   </tr>
   <tr>
      <td></td>
   </tr>
   <tr>
      <td></td>
   </tr>
</table>

- 聚类算法比较

