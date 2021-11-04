from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn. model_selection import train_test_split
clf1=RandomForestClassifier()
X_train,X_test,y_train,y_test=train_test_split(features,lab,shuffle=True,test_size=0.2)
clf1.fit(X_train,y_train)
pred=clf1.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

import pickle
pickle.dump(clf1, open("Yam_three_class.pkl","wb"))
