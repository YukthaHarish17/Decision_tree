# Decision_tree
Let's use the Iris dataset to build a decision tree classifier. We'll start by loading the data and creating a DataFrame for better manipulation. Then, we'll split the data into training and testing sets, build the decision tree model, and visualize the results.

Here's the complete code:

```python
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame for the dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the DataFrame
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Plotting feature importance
feature_importance = clf.feature_importances_
sns.barplot(x=feature_importance, y=iris.feature_names)
plt.title("Feature Importance")
plt.show()
```

### Explanation

1. **Load the dataset**: We load the Iris dataset from `sklearn.datasets`.
2. **Create a DataFrame**: We convert the dataset into a DataFrame for easier data manipulation and visualization.
3. **Split the data**: We split the dataset into training and testing sets.
4. **Train the model**: We create a Decision Tree Classifier and train it using the training data.
5. **Make predictions**: We use the trained model to predict the labels for the test data.
6. **Evaluate the model**: We calculate the accuracy of the model using the test data.
7. **Visualize the decision tree**: We plot the trained decision tree.
8. **Feature importance**: We visualize the importance of each feature in the decision tree model.

This should give you a complete picture of how to use a decision tree classifier with the Iris dataset, including model training, evaluation, and visualization.
