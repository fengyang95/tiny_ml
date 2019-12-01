# tinyml
利用numpy实现的一些周志华《机器学习》（西瓜书）一书及 斯坦福cs229课程中的算法，宜配合西瓜书和cs229课件食用。并选择性实现了一些经典算法的简易版本，
如 按照陈天奇的slides实现的XGBRegressor。
## 已经实现的算法
- **线性模型**  
- [LinearRegression](/tinyml/linear_model/LinearRegression.py)    [线性回归闭式解推导](notes/linear_model/linear_reg_closed_form.pdf)  
- [LogisticRegression](/tinyml/linear_model/LogisticRegression.py)  [逻辑回归相关推导](/notes/linear_model/logistic_regression.pdf)  
- [SGDRegressor](/tinyml/linear_model/SGDRegressor.py)  
- [LocallyWeightedLinearRegression](/tinyml/linear_model/LocallyWeightedLinearRegression.py)    
- **判别分析**  
- [LDA](/tinyml/discriminant_analysis/LDA.py)  
- [GDA](/tinyml/discriminant_analysis/GDA.py)    
- **决策回归树**   
- [DecisionTreeClassifier](/tinyml/tree/DecisionTreeClassifier.py)  
- [DecisionTreeRegressor](/tinyml/tree/DecisionTreeRegressor.py)    
- **支持向量机**  
- [SVC](/tinyml/svm/SVC.py)  
- **贝叶斯**  
- [NaiveBayesClassifier](/tinyml/bayes/NaiveBayesClassifier.py)  
- **聚类算法**
- [KMeans](/tinyml/cluster/KMeans.py)  
- [LVQ](/tinyml/cluster/LVQ.py)  
- [GaussianMixture](/tinyml/cluster/GaussianMixture.py)  
- [DBSCAN](/tinyml/cluster/DBSCAN.py)  
- [AGNES](/tinyml/cluster/AGNES.py)    
- **降维算法**  
- [MDS](/tinyml/dimension_reduction/MDS.py)  
- [PCA](/tinyml/dimension_reduction/PCA.py)  
- [KernelPCA](/tinyml/dimension_reduction/KernelPCA.py)  
- [LLE](/tinyml/dimension_reduction/LLE.py)  
- [Isomap](/tinyml/dimension_reduction/Isomap.py)    
- **集成学习**  
- [AdaBoostClassifier](/tinyml/ensemble/AdaBoostClassifier.py)  
- [GradientBoostingRegressor](/tinyml/ensemble/GradientBoostingRegressor.py)  
- [RandomForestRegressor](/tinyml/ensemble/RandomForestRegressor.py)  
- [XGBRegressor](/tinyml/ensemble/XGBRegressor.py)    
- **特征选择**  
- [ReliefFeatureSelection](/tinyml/feature_selection/ReliefFeatureSelection.py)  
## 和sklearn实现的比较
- **回归算法结果** [代码](/tinyml/compare/compare_regresssor.py)
<table>
    <tr>
        <td rowspan="2">Algorithm vs. RMSE</td>
        <td colspan="2">sklearn-boston</td>
    </tr>
    <tr>
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

- **分类算法结果** [代码](/tinyml/compare/compare_classification.py)
<table>
   <tr>
       <td rowspan="2">Algorithm vs. RMSE</td>
       <td colspan="2">sklearn-breast_cancer</td>
   </tr>
   <tr>
      <td>tinyml</td>
      <td>sklearn</td>
   </tr>
   <tr>
      <td>NaiveBayes</td>
      <td>90.64%</td>
      <td>90.64%</td>
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
</table>

- **聚类算法比较** [代码](/tinyml/compare/compare_clustering.py)
- KMeans
<div align="center">
<img src="/tinyml/compare/cluster_result/tinyml_KMeans.jpg" height="300px" alt="tinyml KMeans" >
<img src="/tinyml/compare/cluster_result/sklearn_KMeans.jpg" height="300px" alt="sklearn KMeans" >
</div>

- DBSCAN
<div align="center">
<img src="/tinyml/compare/cluster_result/tinyml_DBSCAN.jpg" height="300px" alt="tinyml DBSCAN" >
<img src="/tinyml/compare/cluster_result/sklearn_DBSCAN.jpg" height="300px" alt="sklearn DBSCAN" >
</div>

- GMM
<div align="center">
<img src="/tinyml/compare/cluster_result/tinyml_GMM.jpg" height="300px" alt="tinyml GMM" >
<img src="/tinyml/compare/cluster_result/sklearn_GMM.jpg" height="300px" alt="sklearn GMM" >
</div>

- AGNES
<div align="center">
<img src="/tinyml/compare/cluster_result/tinyml_AGNES.jpg" height="300px" alt="tinyml AGNES" >
<img src="/tinyml/compare/cluster_result/sklearn_AGNES.jpg" height="300px" alt="sklearn AGNES" >
</div>

- **降维算法比较** [代码](/tinyml/compare/compare_dimension_reduction.py)
- PCA
<div align="center">
<img src="/tinyml/compare/dimension_reduction_result/tinyml_PCA.jpg" height="300px" alt="tinyml PCA" >
<img src="/tinyml/compare/dimension_reduction_result/sklearn_PCA.jpg" height="300px" alt="sklearn PCA" >
</div>

- KernalPCA
<div align="center">
<img src="/tinyml/compare/dimension_reduction_result/tinyml_KernalPCA.jpg" height="300px" alt="tinyml KernalPCA" >
<img src="/tinyml/compare/dimension_reduction_result/sklearn_KernalPCA.jpg" height="300px" alt="sklearn KernalPCA" >
</div>

- LLE
<div align="center">
<img src="/tinyml/compare/dimension_reduction_result/tinyml_LLE.jpg" height="300px" alt="tinyml LLE" >
<img src="/tinyml/compare/dimension_reduction_result/sklearn_LLE.jpg" height="300px" alt="sklearn LLE" >
</div>

- MDS
<div align="center">
<img src="/tinyml/compare/dimension_reduction_result/tinyml_MDS.jpg" height="300px" alt="tinyml MDS" >
<img src="/tinyml/compare/dimension_reduction_result/sklearn_MDS.jpg" height="300px" alt="sklearn MDS" >
</div>






