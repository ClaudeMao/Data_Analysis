import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

train_set = pd.read_csv('train.csv')
df = pd.DataFrame(train_set)
a = df.describe() #a dataframe give out 'count' and 'mean'
b = df.info()     #a dateframe including data type
#print(a,b)
survived_df = df[df['Survived']==1]
dead_df = df[df['Survived']==0]

#stacked bar chart shown
def drawing():
    #drawing a stacked bar chart 'gender vs survivors'
    labels = ['men','women']
    survived_male = survived_df[survived_df['Sex']=='male']
    survived = [len(survived_male),len(survived_df)-len(survived_male)]
    dead_male = dead_df[dead_df['Sex']=='male']
    dead = [len(dead_male),len(dead_df)-len(dead_male)]
    print(survived,dead)

    width = 0.2

    fig =plt.figure()
    ax = fig.add_subplot(221)

    ax.bar(labels,survived,width,label='Survived')
    ax.bar(labels,dead,width,label='Dead',bottom=survived)

    ax.set_ylabel('Number of people')
    ax.set_title('Gender vs Survival')


    #drawing stacker bar chart 'Pclass vs survivors'
    labels_2 = ['class_1','class_2','class_3']
    survived_class_1 = survived_df[survived_df['Pclass']==1]
    survived_class_2 = survived_df[survived_df['Pclass']==2]
    survived_class_3 = survived_df[survived_df['Pclass']==3]
    survived = [len(survived_class_1),len(survived_class_2),len(survived_class_3)]
    dead_class_1 = dead_df[dead_df['Pclass']==1]
    dead_class_2 = dead_df[dead_df['Pclass']==2]
    dead_class_3 = dead_df[dead_df['Pclass']==3]
    dead = [len(dead_class_1),len(dead_class_2),len(dead_class_3)]

    bx = fig.add_subplot(222)

    bx.bar(labels_2,survived,width,label='Survived')
    bx.bar(labels_2,dead,width,label='Dead',bottom=survived)

    bx.set_ylabel('Number of people')
    bx.set_title('Pclass vs Survival')

    #drawing stacker bar chart 'Embarkment vs survivors'
    labels_3 = ['Southampton','Cherbourg','Queenstown']
    survived_s = survived_df[survived_df['Embarked']=='S']
    survived_c = survived_df[survived_df['Embarked']=='C']
    survived_q = survived_df[survived_df['Embarked']=='Q']
    survived = [len(survived_s),len(survived_c),len(survived_q)]
    dead_s = dead_df[dead_df['Embarked']=='S']
    dead_c = dead_df[dead_df['Embarked']=='C']
    dead_q = dead_df[dead_df['Embarked']=='Q']
    dead = [len(dead_s),len(dead_c),len(dead_q)]

    cx = fig.add_subplot(223)

    cx.bar(labels_3,survived,width,label='Survived')
    cx.bar(labels_3,dead,width,label='Dead',bottom=survived)

    cx.set_ylabel('Number of people')
    cx.set_title('Embarked vs Survival')

    #drawing stacked bar chart 'families vs survivors'
    #make a new dataframe including a new column called 'families = SibSp + Parch'
    new = df.eval('Families = SibSp + Parch',inplace = False)
    labels_4 = ['0','1','2','3','4','5','6','7','8','9','10']
    survived_new = new[new['Survived']==1]
    dead_new = new[new['Survived']==0]
    survived_fam_0 = survived_new[survived_new['Families']==0]
    survived_fam_1 = survived_new[survived_new['Families']==1]
    survived_fam_2 = survived_new[survived_new['Families']==2]
    survived_fam_3 = survived_new[survived_new['Families']==3]
    survived_fam_4 = survived_new[survived_new['Families']==4]
    survived_fam_5 = survived_new[survived_new['Families']==5]
    survived_fam_6 = survived_new[survived_new['Families']==6]
    survived_fam_7 = survived_new[survived_new['Families']==7]
    survived_fam_8 = survived_new[survived_new['Families']==8]
    survived_fam_9 = survived_new[survived_new['Families']==9]
    survived_fam_10 = survived_new[survived_new['Families']==10]
    survived = [len(survived_fam_0),len(survived_fam_1),len(survived_fam_2),\
    len(survived_fam_3),len(survived_fam_4),len(survived_fam_5),len(survived_fam_6),\
    len(survived_fam_7),len(survived_fam_8),len(survived_fam_9),len(survived_fam_10)]
    print(survived)
    dead_fam_0 = dead_new[dead_new['Families']==0]
    dead_fam_1 = dead_new[dead_new['Families']==1]
    dead_fam_2 = dead_new[dead_new['Families']==2]
    dead_fam_3 = dead_new[dead_new['Families']==3]
    dead_fam_4 = dead_new[dead_new['Families']==4]
    dead_fam_5 = dead_new[dead_new['Families']==5]
    dead_fam_6 = dead_new[dead_new['Families']==6]
    dead_fam_7 = dead_new[dead_new['Families']==7]
    dead_fam_8 = dead_new[dead_new['Families']==8]
    dead_fam_9 = dead_new[dead_new['Families']==9]
    dead_fam_10 = dead_new[dead_new['Families']==10]
    dead = [len(dead_fam_0),len(dead_fam_1),len(dead_fam_2),\
    len(dead_fam_3),len(dead_fam_4),len(dead_fam_5),len(dead_fam_6),\
    len(dead_fam_7),len(dead_fam_8),len(dead_fam_9),len(dead_fam_10)]

    dx = fig.add_subplot(224)

    dx.bar(labels_4,survived,width,label='Survived')
    dx.bar(labels_4,dead,width,label='Dead',bottom=survived)

    dx.set_ylabel('Number of people')
    dx.set_title('Families vs Survival')

    ax.legend()
    bx.legend()
    cx.legend()
    dx.legend()
    fig.tight_layout() #automativally adjust the blank part to exclude interacting
    plt.show()

def logistic():
    # fill in blanks in age with mean values and blanks in embarked with most frequent values
    new = df.eval('Families = SibSp + Parch',inplace = False)
    new = new.drop(columns='PassengerId')
    new = new.drop(columns='Name')
    new = new.drop(columns='SibSp')
    new = new.drop(columns='Parch')
    new = new.drop(columns='Ticket')
    new = new.drop(columns='Fare')
    new = new.drop(columns='Cabin')
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    most_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_age = new['Age'].values.reshape(-1,1)
    new_embark = new['Embarked'].values.reshape(-1,1)
    new['Age']=mean_imputer.fit_transform(new_age)
    new['Embarked']=most_imputer.fit_transform(new_embark)
    new.info() #info of filled dataset
    # dummy encoded
    new = pd.get_dummies(new,columns=['Sex','Embarked'])
    new.info()
    '''
    data preprocessing accomplished
    next, we gonna try logistic regression for titanic survival prediction
    apparently, ticket number, cabin and fare have barely connection with survival
    this gives us an intuition of feature selection
    '''
    data = new.loc[:,new.columns[1]:new.columns[8]]
    print(data)
    
    result = new['Survived']
    result = np.array(new['Survived']).reshape(-1,1)
    print(data.shape,result.shape)

    X_train, X_test, y_train, y_test = train_test_split(data,result,train_size=0.8)
    clf = LogisticRegression(solver='liblinear',max_iter=1000)
    clf.fit(X_train, y_train.ravel()) #following the warning given by terminal
    print('mean accuracy on the training data :{:.3f}'.format(clf.score(X_train, y_train)))
    print('mean accuracy on the test data :{:.3f}'.format(clf.score(X_test, y_test)))
    
    test = pd.read_csv('test.csv')
    test_df = pd.DataFrame(test)
    new = test_df.eval('Families = SibSp + Parch',inplace = False)
    new = new.drop(columns='PassengerId')
    new = new.drop(columns='Name')
    new = new.drop(columns='SibSp')
    new = new.drop(columns='Parch')
    new = new.drop(columns='Ticket')
    new = new.drop(columns='Fare')
    new = new.drop(columns='Cabin')
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    most_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_age = new['Age'].values.reshape(-1,1)
    new_embark = new['Embarked'].values.reshape(-1,1)
    new['Age']=mean_imputer.fit_transform(new_age)
    new['Embarked']=most_imputer.fit_transform(new_embark)
    new.info() #info of filled dataset
    # dummy encoded
    new = pd.get_dummies(new,columns=['Sex','Embarked'])
    new.info()

    data = new.loc[:,new.columns[0]:new.columns[7]]
    predict_result = clf.predict(data)
    #predict_result = np.array(predict_result).reshape(-1,1)
    print(predict_result)
    
    submission = test_df['PassengerId']
    submission = pd.DataFrame(submission)
    print(submission)
    submission.insert(1,'Survived', predict_result)
    print(submission)
    outputpath = 'submission.csv'
    submission.to_csv(outputpath,index=False)
    
def random_forest():
    new = df.eval('Families = SibSp + Parch',inplace = False)
    new = new.drop(columns='PassengerId')
    new = new.drop(columns='Name')
    new = new.drop(columns='SibSp')
    new = new.drop(columns='Parch')
    new = new.drop(columns='Ticket')
    new = new.drop(columns='Fare')
    new = new.drop(columns='Cabin')
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    most_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_age = new['Age'].values.reshape(-1,1)
    new_embark = new['Embarked'].values.reshape(-1,1)
    new['Age']=mean_imputer.fit_transform(new_age)
    new['Embarked']=most_imputer.fit_transform(new_embark)

    new = pd.get_dummies(new,columns=['Sex','Embarked'])

    data = new.loc[:,new.columns[1]:new.columns[8]]
    
    result = new['Survived']
    result = np.array(new['Survived']).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(data,result,train_size=0.6)
    clf = RandomForestClassifier(n_estimators=5000,max_features=None,max_depth=20)

    clf.fit(X_train, y_train.ravel()) #following the warning given by terminal
    print('mean accuracy on the training data :{:.3f}'.format(clf.score(X_train, y_train)))
    print('mean accuracy on the test data :{:.3f}'.format(clf.score(X_test, y_test)))
    #print(clf.feature_importances_)
    #feature_weights=sorted(zip(map(lambda x: round(x, 2), clf.feature_importances_), data.columns), reverse=True)
    #print(feature_weights)

    test = pd.read_csv('test.csv')
    test_df = pd.DataFrame(test)
    new = test_df.eval('Families = SibSp + Parch',inplace = False)
    new = new.drop(columns='PassengerId')
    new = new.drop(columns='Name')
    new = new.drop(columns='SibSp')
    new = new.drop(columns='Parch')
    new = new.drop(columns='Ticket')
    new = new.drop(columns='Fare')
    new = new.drop(columns='Cabin')
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    most_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_age = new['Age'].values.reshape(-1,1)
    new_embark = new['Embarked'].values.reshape(-1,1)
    new['Age']=mean_imputer.fit_transform(new_age)
    new['Embarked']=most_imputer.fit_transform(new_embark)
    # dummy encoded
    new = pd.get_dummies(new,columns=['Sex','Embarked'])

    data = new.loc[:,new.columns[0]:new.columns[7]]
    predict_result = clf.predict(data)
    submission = test_df['PassengerId']
    submission = pd.DataFrame(submission)

    submission.insert(1,'Survived', predict_result)

    outputpath = 'random_forest.csv'
    submission.to_csv(outputpath,index=False)
    #0.799
    
def decision_tree():
    new = df.eval('Families = SibSp + Parch',inplace = False)
    new = new.drop(columns='PassengerId')
    new = new.drop(columns='Name')
    new = new.drop(columns='SibSp')
    new = new.drop(columns='Parch')
    new = new.drop(columns='Ticket')
    new = new.drop(columns='Fare')
    new = new.drop(columns='Cabin')
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    most_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_age = new['Age'].values.reshape(-1,1)
    new_embark = new['Embarked'].values.reshape(-1,1)
    new['Age']=mean_imputer.fit_transform(new_age)
    new['Embarked']=most_imputer.fit_transform(new_embark)

    new = pd.get_dummies(new,columns=['Sex','Embarked'])

    data = new.loc[:,new.columns[1]:new.columns[8]]
    
    result = new['Survived']
    result = np.array(new['Survived']).reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(data,result,train_size=0.7)
    clf = DecisionTreeClassifier(criterion = 'gini',max_depth = 4,max_features= None, \
        min_samples_leaf= 5, min_samples_split= 2,random_state = 0, splitter ='best')
    clf.fit(X_train, y_train.ravel()) #following the warning given by terminal
    print('mean accuracy on the training data :{:.3f}'.format(clf.score(X_train, y_train)))
    print('mean accuracy on the test data :{:.3f}'.format(clf.score(X_test, y_test)))
    
    test = pd.read_csv('test.csv')
    test_df = pd.DataFrame(test)
    new = test_df.eval('Families = SibSp + Parch',inplace = False)
    new = new.drop(columns='PassengerId')
    new = new.drop(columns='Name')
    new = new.drop(columns='SibSp')
    new = new.drop(columns='Parch')
    new = new.drop(columns='Ticket')
    new = new.drop(columns='Fare')
    new = new.drop(columns='Cabin')
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    most_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_age = new['Age'].values.reshape(-1,1)
    new_embark = new['Embarked'].values.reshape(-1,1)
    new['Age']=mean_imputer.fit_transform(new_age)
    new['Embarked']=most_imputer.fit_transform(new_embark)
    # dummy encoded
    new = pd.get_dummies(new,columns=['Sex','Embarked'])

    data = new.loc[:,new.columns[0]:new.columns[7]]
    predict_result = clf.predict(data)
    
    submission = test_df['PassengerId']
    submission = pd.DataFrame(submission)

    submission.insert(1,'Survived', predict_result)

    outputpath = 'decision_tree.csv'
    submission.to_csv(outputpath,index=False)
    
#drawing()
#logistic()
random_forest()
#decision_tree()