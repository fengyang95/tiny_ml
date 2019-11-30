from sklearn.metrics import mean_squared_error
def train_and_eval(data,regressor):
    train_X, train_y, test_X, test_y=data
    regressor.fit(train_X,train_y)
    preds_y=regressor.predict(test_X)
    mse=mean_squared_error(test_y,preds_y)
    del regressor
    return mse

from sklearn.datasets import load_boston,load_diabetes
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from tinyml.linear_model.LinearRegression import LinearRegression as tinymlLinearRegression
from tinyml.linear_model.SGDRegressor import SGDRegressor as tinymlSGDRegressor
from tinyml.ensemble.GradientBoostingRegressor import GradientBoostingRegressor as tinymlGradientBoostingRegressor
from tinyml.ensemble.RandomForestRegressor import RandomForestRegressor as tinymlRandomForestRegressor
from tinyml.ensemble.XGBRegressor import XGBRegressor as tinymlXGBRegressor
from tinyml.tree.DecisionTreeRegressor import DecisionTreeRegressor as tinymlDecisionTreeRegressor

from sklearn.linear_model import LinearRegression as sklearnLinearRegression
from sklearn.linear_model import SGDRegressor as sklearnSGDRegressor
from sklearn.tree import DecisionTreeRegressor as sklearnDecisonTreeRegressor
from sklearn.ensemble import RandomForestRegressor as sklearnRnadomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as sklearnGradientBoostRegressor
from xgboost import XGBRegressor

if __name__=='__main__':


    boston_X,boston_y=load_boston(return_X_y=True)

    boston_train_X,boston_test_X,boston_train_y,boston_test_y=train_test_split(boston_X,boston_y,test_size=0.3,random_state=0)

    data=boston_train_X,boston_train_y,boston_test_X,boston_test_y

    rmse_tinyml_linear_regression=train_and_eval(data,tinymlLinearRegression())
    print('tinyml LinearRegression:',rmse_tinyml_linear_regression)
    rmse_sklearn_linear_regression=train_and_eval(data,sklearnLinearRegression())
    print('sklearn LinearRegression:',rmse_sklearn_linear_regression)
    print('\n')
    std_scaler=StandardScaler()
    std_scaler.fit(boston_train_X)
    X_train=std_scaler.transform(boston_train_X)
    X_test=std_scaler.transform(boston_test_X)
    rmse_tinyml_sgd_regressor=train_and_eval((X_train,boston_train_y,X_test,boston_test_y),tinymlSGDRegressor(max_iter=200,penalty='l1',alpha=1e-3,l1_ratio=0.5))
    print('tinyml SGDRegressor:',rmse_tinyml_sgd_regressor)
    rmse_sklearn_sgd_regressor=train_and_eval((X_train,boston_train_y,X_test,boston_test_y),sklearnSGDRegressor(max_iter=200,penalty='l1',alpha=1e-3))
    print('sklearn SGDRegressor:',rmse_sklearn_sgd_regressor)
    print('\n')
    rmse_tinyml_decision_tree_regressor=train_and_eval(data,tinymlDecisionTreeRegressor(min_samples_split=20,min_samples_leaf=5))
    print('tinyml DecisionTreeRegressor:',rmse_tinyml_decision_tree_regressor)
    rmse_sklearn_decision_tree_regressor=train_and_eval(data,sklearnDecisonTreeRegressor(min_samples_split=20,min_samples_leaf=5,random_state=False))
    print('sklearn DecisionTreeRegressor:',rmse_sklearn_decision_tree_regressor)
    print('\n')
    rmse_tinyml_random_forest_tree_regressor = train_and_eval(data, tinymlRandomForestRegressor(
            base_estimator=tinymlDecisionTreeRegressor,
            n_estimators=100, min_samples_leaf=5, min_samples_split=15))
    print('tinyml RandomForestRegressor:', rmse_tinyml_random_forest_tree_regressor)
    rmse_sklearn_random_forest_tree_regressor=train_and_eval(data,sklearnRnadomForestRegressor(n_estimators=100, min_samples_leaf=5, min_samples_split=15, random_state=False))
    print('sklearn RandomForestRegressor:',rmse_tinyml_random_forest_tree_regressor)

    rmse_tinyml_gradient_boost_regressor = train_and_eval(data,
                                                          tinymlGradientBoostingRegressor(n_estimators=500,
                                                                                          base_estimator=tree.DecisionTreeRegressor(
                                                                                              min_samples_split=15,
                                                                                              min_samples_leaf=5,
                                                                                              random_state=False)))

    print('tinyml GradientBoostRegressor:', rmse_tinyml_gradient_boost_regressor)
    rmse_sklearn_gradient_boost_regressor=train_and_eval(data,
                                                         sklearnGradientBoostRegressor(n_estimators=500,min_samples_leaf=5,min_samples_split=15,random_state=False))
    print('sklearn GradientBoostRegressor:',rmse_sklearn_gradient_boost_regressor)

    rmse_tinyml_xgbregressor = train_and_eval(data,
                                              tinymlXGBRegressor(n_estimators=100, max_depth=3, gamma=0.))
    print('tinyml XGBRegressor:', rmse_tinyml_xgbregressor)
    rmse_xgboost_xgbregressor=train_and_eval(data,XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=100,gamma=0,reg_lambda=1))
    print('xgboost XGBRegressor:',rmse_xgboost_xgbregressor)



