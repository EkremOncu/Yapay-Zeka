import pandas as pd
import numpy as np
import csv
import random

# Random veri üretimi için seed
np.random.seed(42)
random.seed(42)

# Satır sayısı
num_rows = 10000

# Sütun verileri oluşturuluyor
data = {
    'Yaş': np.random.randint(18, 70, size=num_rows),
    'Gelir': np.random.randint(30000, 100000, size=num_rows),
    'Harcamalar': np.random.randint(10000, 40000, size=num_rows),
    'Kredi_Skoru': np.random.randint(600, 750, size=num_rows),
    'İnternet_Aboneliği': np.random.randint(1, 25, size=num_rows),
    'Yorum': [random.choice(["Harika ürünler!", "Güzel hizmet, ama daha iyi olabilir.", 
         "Beklentilerimin altında.", "Yine de memnun kaldım.", "Ürünler kaliteli, ama fiyatlar yüksek.", 
         "Yeterli bir deneyim.", "İnternetten daha iyi bekliyordum.", "Hizmet çok iyi, teşekkürler.", 
         "Beklediğimden daha iyi.", "Kaliteli, ama fiyat biraz yüksek."]) 
         for _ in range(num_rows)],
    'Puan': np.random.randint(0, 1000, size=num_rows),
    }

# Veri çerçevesi oluşturuluyor
df = pd.DataFrame(data)

# CSV dosyasına yazdırma
df.to_csv('dataset.csv', quoting=csv.QUOTE_NONNUMERIC, index=False)

print("Veri kümesi 'dataset.csv' olarak kaydedildi.")
