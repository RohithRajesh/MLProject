import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.model_selection
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import sklearn.ensemble
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import copy
import sklearn.neural_network

SEED = 40

class preprocessor():
    def __init__(self):
        ''' This functions reads and stores the data'''
        
        # Removed columns having missing values 
        df = pd.read_csv("communities.data",header=None)
        df = df._get_numeric_data()
        df = df.drop([0,4],axis = 1)

        dfs = np.split(df,[-1],axis=1)

        self.x = dfs[0]**(0.5)
        self.y = dfs[1]
        self.y.columns = ["Crime Rate"]

        self.discretization_types=['ei','ep','km']


        # PCA
        pca = sklearn.decomposition.PCA(0.9)
        pca.fit(self.x)
        self.x = pd.DataFrame(pca.transform(self.x))

        self.x_col = self.x.shape[1]

        self.data = self.x.join(self.y)

    def discretization(self,k):
        ''' This function discretizes the target value into k bins using the 3 methods namely EI, EP and KM '''
        # Equal intervals

        ei_discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')


        self.data['ei'] = ei_discretizer.fit_transform(np.array(self.data['Crime Rate']).reshape(-1,1))

        ei_class_value = {}

        for i in range(k):
            filtered = self.data[self.data['ei']==i]
            ei_class_value[i] = np.average(filtered['Crime Rate'])
            

        # Equal Probablities


        ep_discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='quantile')

        self.data['ep'] = ep_discretizer.fit_transform(np.array(self.data['Crime Rate']).reshape(-1,1))


        ep_class_value = {}

        for i in range(k):
            filtered = self.data[self.data['ep']==i]
            ep_class_value[i] = np.average(filtered['Crime Rate'])
            


        # Kmeans

        km_discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans')

        self.data['km'] = km_discretizer.fit_transform(np.array(self.data['Crime Rate']).reshape(-1,1))

        km_class_value = {}

        for i in range(k):
            filtered = self.data[self.data['km']==i]
            km_class_value[i] = np.median(filtered['Crime Rate'])

        self.class_encodings = {}
        self.class_encodings['ei'] = ei_class_value
        self.class_encodings['ep'] = ep_class_value
        self.class_encodings['km'] = km_class_value

        # (self.data['km']).hist()
        # plt.show()
        # (self.data['ei']).hist()
        # plt.show()
        # (self.data['ep']).hist()
        # plt.show()

        return self.class_encodings

    def getData(self,d_type='ei',output=False):
        ''' This function returns the data with the required discretization type'''
        if output:
            return self.x,self.data[d_type],self.data["Crime Rate"]
        return self.x, self.data[d_type]

    def k_fold(self, estimator, d_type = 'ei', folds=5):
        ''' This function performs k folds on a given discretization type and returns different metrics '''
        skf = sklearn.model_selection.StratifiedKFold(n_splits=folds, random_state=SEED, shuffle=True)

        X = np.array(self.x)
        y = self.data[d_type]
        real_values = np.array(self.y)

        train_Accuracy_kfold_list = []
        train_Mae_kfold_list = []
        train_Mse_kfold_list = []  
        train_R2_kfold_list = []  


        Accuracy_kfold_list = []
        Mae_kfold_list = []
        Mse_kfold_list = []  
        R2_kfold_list = []

        # Performing K folds
        for train_index, test_index in skf.split(self.x, self.data[d_type]):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                real_train, real_test = real_values[train_index], real_values[test_index]

                # Training the model
                estimator.fit(X_train, y_train)

                #  Predicting testing and training data
                pred_test = estimator.predict(X_test)
                pred_train = estimator.predict(X_train)

                # Calculating Accuracy
                Accuracy_kfold_list.append(sklearn.metrics.accuracy_score(y_test,pred_test))
                
                # Converting the Classes to their Class marks
                for key in self.class_encodings[d_type].keys():
                    pred_test[pred_test==key] = self.class_encodings[d_type][key]
            
                # Calculating metrics used on continuous data
                Mae_kfold_list.append(sklearn.metrics.mean_absolute_error(real_test,pred_test))

                Mse_kfold_list.append(sklearn.metrics.mean_squared_error(real_test,pred_test))    

                R2_kfold_list.append(sklearn.metrics.r2_score(real_test,pred_test))

                # Calculating Accuracy
                train_Accuracy_kfold_list.append(sklearn.metrics.accuracy_score(y_train,pred_train))

                # Converting the Classes to their Class marks
                for key in self.class_encodings[d_type].keys():
                    pred_train[pred_train==key] = self.class_encodings[d_type][key]

                # Calculating metrics used on continuous data
                train_Mae_kfold_list.append(sklearn.metrics.mean_absolute_error(real_train,pred_train))

                train_Mse_kfold_list.append(sklearn.metrics.mean_squared_error(real_train,pred_train))    
        
                train_R2_kfold_list.append(sklearn.metrics.r2_score(real_train,pred_train))


        # Averaging the metrics over folds 
        Accuracy_kfold = sum(Accuracy_kfold_list)/len(Accuracy_kfold_list)
        Mse_kfold = sum(Mse_kfold_list)/len(Mse_kfold_list)
        Mae_kfold = sum(Mae_kfold_list)/len(Mae_kfold_list)
        R2_kfold = sum(R2_kfold_list)/len(R2_kfold_list)

        train_Accuracy_kfold = sum(train_Accuracy_kfold_list)/len(train_Accuracy_kfold_list)
        train_Mse_kfold = sum(train_Mse_kfold_list)/len(train_Mse_kfold_list)
        train_Mae_kfold = sum(train_Mae_kfold_list)/len(train_Mae_kfold_list)
        train_R2_kfold = sum(train_R2_kfold_list)/len(train_R2_kfold_list)
        
        return Accuracy_kfold, Mae_kfold, Mse_kfold,R2_kfold, train_Accuracy_kfold, train_Mae_kfold, train_Mse_kfold,train_R2_kfold


p = preprocessor()

p.discretization(10)

for d_type in ['ei','ep','km']:
    save_x,save_label,save_value = p.getData(d_type,True)
    save_data = copy.deepcopy(save_x)
    save_data['label']=save_label
    save_data['Crime Rate']=save_value

    save_data.to_csv('./Data/10_'+d_type+".csv",index=False)



def classwise_graphs(p):

    d_classes = list(range(5,21,5))

    d_classes = [2,3,4] + d_classes

    d_acc = {}
    d_mse = {}
    d_mae = {}
    d_r2 = {}

    dt_acc = {}
    dt_mse = {}
    dt_mae = {}
    dt_r2 = {}

    # Loop iterating on types of discretization methods
    for dis_type in ['ep','km','ei']:
        accuracies = []
        mses = []
        maes = []
        r2s = []
        
        t_accuracies = []
        t_maes = []
        t_mses = []
        t_r2s = []

        # Loop over number of bins
        for bins in d_classes:
            # print(bins)
            p.discretization(bins)

            # model = sklearn.naive_bayes.GaussianNB()
            model = sklearn.svm.SVC(kernel='linear')
            # model = sklearn.ensemble.RandomForestClassifier()
            # model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=[8,4],solver="sgd",learning_rate="adaptive",learning_rate_init=1,max_iter=1000, shuffle=False)
            
            accuracy, mae, mse, r2, t_accuracy, t_mae, t_mse, t_r2 =  p.k_fold(model, d_type=dis_type)
            
            accuracies.append(accuracy)
            mses.append(mse)
            maes.append(mae)
            r2s.append(r2)

            t_accuracies.append(t_accuracy)
            t_maes.append(t_mae)
            t_mses.append(t_mse)
            t_r2s.append(t_r2)
        print("Accuracy", accuracies)
        print("MSE", mses)
        print("Mae",maes)
        print("R2",r2s)
        
        print("Train Accuracy", t_accuracies)
        print("Train MSE", t_mses)
        print("Train Mae",t_maes)
        print("Train R2",t_r2s)


        d_acc[dis_type] = accuracies
        d_mae[dis_type] = maes
        d_mse[dis_type] = mses
        d_r2[dis_type] = r2s

        dt_acc[dis_type] = t_accuracies
        dt_mae[dis_type] = t_maes
        dt_mse[dis_type] = t_mses
        dt_r2[dis_type] = t_r2s

    m_name = "svm"

    plt.figure(0)
    plt.plot(d_classes, d_acc['ei'], label='Equal Intervals')
    plt.plot(d_classes, d_acc['ep'], label='Equal Probability')
    # plt.plot(d_classes, dt_acc['ep'], label='Training Equal Probability')
    plt.plot(d_classes, d_acc['km'], label='K Means Clustering')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_acc.png")

    plt.figure(1)
    plt.plot(d_classes, d_mse['ei'], label='Equal Intervals')
    plt.plot(d_classes, d_mse['ep'], label='Equal Probability')
    # plt.plot(d_classes, dt_mse['ep'], label='Training Equal Probability')
    plt.plot(d_classes, d_mse['km'], label='K Means Clustering')
    plt.title('MSE')
    plt.ylabel('MSE')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_mse.png")

    plt.figure(2)
    plt.plot(d_classes, d_mae['ei'], label='Equal Intervals')
    plt.plot(d_classes, d_mae['ep'], label='Equal Probability')
    # plt.plot(d_classes, dt_mae['ep'], label='Training Equal Probability')
    plt.plot(d_classes, d_mae['km'], label='K Means Clustering')
    plt.title('MAE')
    plt.ylabel('MAE')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_mae.png")

    plt.figure(3)
    plt.plot(d_classes, d_r2['ei'], label='Equal Intervals')
    plt.plot(d_classes, d_r2['ep'], label='Equal Probability')
    # plt.plot(d_classes, dt_r2['ep'], label='Training Equal Probability')
    plt.plot(d_classes, d_r2['km'], label='K Means Clustering')
    plt.title('R2 Score')
    plt.ylabel('R2 Score')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_r2.png")

# classwise_graphs(p)




def hidden_units(p):
    ''' This function plots the graph of metrics vs number of hidden units '''
    d_acc = []
    d_mse = []
    d_mae = []
    d_r2 = []

    dt_acc = []
    dt_mse = []
    dt_mae = []
    dt_r2 = []

    # Discretization type considered
    dis_type = 'ep'
    h_units = range(1,17)
    # Loop over hidden units
    for hu in h_units:

        print(hu)
        p.discretization(10)
        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=[hu],solver="sgd",learning_rate="adaptive",learning_rate_init=1,max_iter=1000, shuffle=False)
        
        accuracy, mae, mse, r2, t_accuracy, t_mae, t_mse, t_r2 =  p.k_fold(model, d_type=dis_type)
        
        d_acc.append(accuracy)
        d_mse.append(mse)
        d_mae.append(mae)
        d_r2.append(r2)

        dt_acc.append(t_accuracy)
        dt_mae.append(t_mae)
        dt_mse.append(t_mse)
        dt_r2.append(t_r2)


    m_name = "nn_4_4"

    plt.figure(0)
    plt.plot(h_units, d_acc, label='Validation')
    plt.plot(h_units, dt_acc, label='Training')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_acc.png")

    plt.figure(1)
    plt.plot(h_units, d_mse, label='Validation')
    plt.plot(h_units, dt_mse, label='Training')
    plt.title('MSE')
    plt.ylabel('MSE')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_mse.png")

    plt.figure(2)
    plt.plot(h_units, d_mae, label='Validation')
    plt.plot(h_units, dt_mae, label='Training')
    plt.title('MAE')
    plt.ylabel('MAE')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_mae.png")

    plt.figure(3)
    plt.plot(h_units, d_r2, label='Equal Probability')
    plt.plot(h_units, dt_r2, label='Training Equal Probability')
    plt.title('R2 Score')
    plt.ylabel('R2 Score')
    plt.xlabel('Number of classes')
    plt.legend()
    plt.savefig(m_name+"_r2.png")

hidden_units(p)