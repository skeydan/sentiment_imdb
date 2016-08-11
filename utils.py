from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def assess_classification_performance(model, X_train, y_train, X_test, y_test, short = False):
  
    accuracy_train = accuracy_score(y_train, model.predict(X_train))
    accuracy_test = accuracy_score(y_test, model.predict(X_test))
    print('\nClassification performance overview:\n************************************')
    print('accuracy (train/test): {} / {}\n'.format(accuracy_train, accuracy_test))
    
    if not short:
    
      # confusion matrix
      # rows: actual group
      # columns: predicted group
      print('Confusion_matrix (training data):')
      print(confusion_matrix(y_train, model.predict(X_train)))
      
      print('Confusion_matrix (test data):')
      print(confusion_matrix(y_test, model.predict(X_test)))

      # precision =  tp / (tp + fp)
      # recall = tp / (tp + fn) (= sensitivity)
      # F1 = 2 * (precision * recall) / (precision + recall)
      print('\nPrecision - recall (training data):')
      print(classification_report(y_train, model.predict(X_train)))
      
      print('\nPrecision - recall (test data):')
      print(classification_report(y_test, model.predict(X_test)))

