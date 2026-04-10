from sklearn.tree import DecisionTreeClassifier

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model.fit(X_train, y_train)
    return model
