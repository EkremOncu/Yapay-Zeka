from sklearn.datasets import make_classification

dataset, labels = make_classification(100, 10, n_classes=4, n_informative=4)

print(dataset)
print(labels)
