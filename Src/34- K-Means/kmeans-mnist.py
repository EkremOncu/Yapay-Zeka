NCLUSTERS = 10

from tensorflow.keras.datasets import mnist

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = mnist.load_data()

scaled_training_dataset_x  = training_dataset_x.reshape(-1, 28 * 28) / 255
scaled_test_dataset_x  = test_dataset_x.reshape(-1, 28 * 28) / 255


from sklearn.cluster import KMeans

km = KMeans(n_clusters=NCLUSTERS, n_init=10)
km.fit(scaled_training_dataset_x)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
transformed_dataset_x = pca.fit_transform(scaled_training_dataset_x)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.title('Clustered Points')
for i in range(NCLUSTERS):
    plt.scatter(transformed_dataset_x[km.labels_ == i, 0], transformed_dataset_x[km.labels_ == i, 1])
plt.show()


# etiketlendirmenin örnek sonuçları

import matplotlib.pyplot as plt


for cluster_no in range(10):
    plt.figure(figsize=(6, 10))
    print(f'Cluster No: {cluster_no}')
    for i, picture in enumerate(training_dataset_x[km.labels_ == cluster_no][:50]):
        plt.subplot(13, 5, i + 1)
        plt.imshow(picture, cmap='gray')
    plt.show()
    print('-' * 30)

