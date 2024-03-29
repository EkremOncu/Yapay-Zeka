"""
---------------------------------------------------------------------------
C ve Sistem Programcıları Derneği Sınıfta Yapılan Örnekler ve Özet Notlar
                                
Bu notlar Kaan ASLAN'ın notlarından yararlanılarak oluşturulmuştur. 
---------------------------------------------------------------------------
"""

#  ----------------------------- NumPy  -----------------------------
"""
------------------------------------------------------------------------------------
Python'a vektörel işlem yapma yeteneğini kazandırmak için bazı üçüncü parti kütüphaneler 
oluşturulmuştur. Bunların en çok kullanılanı "NumPy" isimli kütüphanedir. Numpy 
C'de yazılmış bir kütüphanedir.

!!! NumPy dizileri  MUTABLE 'dir. (değiştirilebilir)
------------------------------------------------------------------------------------

import numpy  as np

a = np.array([  [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
print(a, type(a))
------------------------------------------------------------------------------------

Anımsanacağı gibi Python'daki list, tuple, gibi veri yapıları aslında değerlerin i
kendisini tutmamaktadır. Değerlerin tutulduğu nesnelerin adreslerini tutmaktadır. 
Bu biçimdeki çalışma yoğun sayısal işlemlerde oldukça hantal hale gelmektedir. Bu 
nedenle NumPy dizileri değerleri Python'un listeleri gibi değil C Programlama 
Dilindeki diziler gibi tutmaktadır. Yani NumPy dizilerinin elemanları genel olarak 
aynı türdendir.Örneğin:

a = np.array([1, 2, 3, 4, 5])
a.dtype

Burada a değişkenin gösteridği NumPy dizisi tamamen C'deki gibi bir dizidir. Yani 
NumPy dizisinin elemanları adresleri değil doğrudan değerlerin kendisini tutmaktadır. 
Bu nedenle NumPy dizileri (yani ndarray nesneleri) birkaç istisna durum dışında 
homojendir.
------------------------------------------------------------------------------------

array fonksiyonuyla NumPy dizisi (ndarray nesnesi) yaratılırken yaratılacak NumPy 
dizisinin C'deki dtype türü array fonksiyonun dtype parametresi ile açıkça 
belirlenebilir.

a = np.array([1, 2, 3, 4, 5], dtype= 'float64' )
a = np.array([1, 2, 3, 4, 5], dtype= np.float64 )
#  array([1., 2., 3., 4., 5.])

 NumPy dizilerinde kullanabileceğimiz dtype türlerinin önemli olanları şunlardır:  

bool_ /bool8        : bool türü
byte / int8         : bir byte'lık işaretli tamsayı türü
ubyte / uint8       : bir byte'lık işaretsiz tamsayı türü
short / int16       : iki byte'lık işaretli tamsayı türü
ushort / uint16     : iki byte'lık işaretsiz tamsayı türü
int32               : dört byte'lık işaretli tamsayı türü
uint32              : dört byte'lık işaretsiz tamsayı türü
int64               : sekiz byte2lık işaretli tamsayı türü
uint64              : sekiz byte'lık işaretsiz tamsayı türü
float32 / single    : dört byte'lık gerçek sayı türü

float64 / float / double : sekiz byte'lık gerçek sayı türü


işaretli tamsayı -> Örneğin : (-128, 127)
işaretsiz tamsayı -> Örneğin : (0, 255)

-> işaretli tamsayı = bitlerden biri işarete ayrılmış demek

!!! Bir NumPy dizisinin dtype'ı demek -> Onun C dilindeki gerçek(orjinal) türü demek
------------------------------------------------------------------------------------

String sınıfı C'deki string sınıfı gibi davranır, iterable değil.

import numpy  as np
a = np.array('ankara')
print(a)

a = np.array(list('ankara'))
print(a)
------------------------------------------------------------------------------------

İçi sıfırlarla dolu numpy dizilerinin oluşturulması gerekebilmektedir. Bunun için 
zeros fonksiyonu kullanılmaktadır. zeros fonksiyonun yine birinci parametresi 
oluşturulacak Numpy dizisinin boyutlarını (shape) belirtir.
 
import numpy  as np
a = np.zeros(10, dtype='int8') # tek boyutlu dizi
print(a)
print()

a = np.zeros( (10,10) , dtype='int8') # iki boyutlu dizi
print(a)
------------------------------------------------------------------------------------

ones isimli fonksiyon içi 1'lerle dolu bir NumPy dizisi oluşturmaktadır. Yine fonksiyonun birinci parametresi oluşturulacak dizinin boyutlarını 
belirtir. dtype parametresi ise dtype türünü belirtir.  Örneğin:

a = np.ones(10, dtype='int32')
# array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
------------------------------------------------------------------------------------

full isimli fonksiyon NumPy dizisini bizim istediğimiz değerle doldurarak yaratır. 
(Yani zeros ve ones fonksiyonlarının genel biçimidir.) Bu fonksiyonun yine birinci 
parametresi NumPy dizisinin boyutlarını, ikinci parametresi doldurulacak değerleri 
belirtmektedir. Fonksiyonda yine dtype belirtilebilir.

a = np.full(10, 5, dtype='int8')
# array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=int8)

b = np.full((5, 5), 1.2, dtype='float32')
print(b)
------------------------------------------------------------------------------------
"""

# ---------- random ------------
"""
------------------------------------------------------------------------------------

Rastgele değerlerden NumPy dizisi oluşturabilmek için numpy.random modülünde çeşitli fonksiyonlar bulundurulmuştur. Örneğin 
numpy.random.random fonksiyonu belli bir boyutta 0 ile 1 arasında rastgele gerçek sayı değerleri oluşturmaktadır. Bu fonksiyon dtype
parametresine sahip değildir. Her zaman float64 olarak numpy dizisini yaratmaktadır.Fonksiyonun boyut belirten bir parametresi vardır
------------------------------------------------------------------------------------
import numpy  as np

a = np.random.random(5)
#  array([0.16421352, 0.73812732, 0.78583484, 0.37765506, 0.25084917])

a = np.random.random((5, 5))
# array([[0.90589859, 0.41192256, 0.48235128, 0.57842162, 0.50297967],
       [0.98173552, 0.80423981, 0.4047245 , 0.85172597, 0.32276178],
       [0.85301294, 0.26888055, 0.14672257, 0.67893162, 0.06198328],
       [0.80185088, 0.52359397, 0.35513698, 0.84171051, 0.34803395],
       [0.15082236, 0.20827871, 0.70286441, 0.46469883, 0.18069756]])

------------------------------------------------------------------------------------
numpy.random.randint fonksiyonu [low, high) aralığında rastgele tamsayı 
değerlerinden oluşan NumPy dizisi oluşturmaktadır
                                 
a = np.random.randint(10, 20, (5, 5), dtype='int8')
# array([[15, 14, 12, 13, 14],
       [16, 13, 13, 10, 12],
       [12, 15, 15, 15, 19],
       [18, 13, 13, 14, 13],
       [14, 12, 18, 19, 14]], dtype=int8)
------------------------------------------------------------------------------------
"""

# ---------- arange ------------
"""
------------------------------------------------------------------------------------

arange fonksiyonu Python'ın built-in range fonksiyonuna benzemektedir. Ancak arange 
bize dolaşılabilir bir nesne vermez. Doğrudan bir NumPy dizisi verir. start, stop, 
step parametreleri range fonksiyonunda olduğu gibidir. Ancak Python range fonksiyonunda 
start, stop ve step değerleri int türünden olmak zorundayken arange fonksiyonunda 
float türünden de oabilir. Böylelikle biz arange ile noktasal artırımlarla bir 
numpy dizisi oluşturabiliriz. arange fonksiyonu dtype parametresi de alabilmektedir. 

Ancak bu fonksiyon her zaman tek boyutlu bir diziyi bize verir.
------------------------------------------------------------------------------------

import numpy  as np
a = np.arange(-1, 1, 1)
# array([-1,  0])

a = np.arange(-1, 1, 0.25)
#  array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75])
------------------------------------------------------------------------------------

arange fonksiyonunu kullanırken dikkat etmek gerekir. Çünkü noktasal artırımlar, 
noktasal start ve stop değerleri yuvarlama hatalarından dolayı beklenenden fazla 
ya da az sayıda eleman üretebilir. (Örneğin 0.1 artırımlarla ilerleken yuvarlama 
hatasından dolayı stop değerine çok yakın ama ondan küçük değer elde edilebilir 
ve bu değer de dizi içinde bulunabilir.) Zaten Python'daki built-in range sınıfının 
tamsayı değerler almasının nedeni de budur. 
------------------------------------------------------------------------------------
"""

# ---------- linspace ------------
"""
------------------------------------------------------------------------------------
arange fonksiyonun yukarıda belirtilen probleminden dolayı noktasal artırım için 
genellikle programcılar linspace fonksiyonunu tercih ederler. Bu fonksiyon start, 
stop ve num parametrelerine sahiptir. Fonksiyon her zaman start ve stop değerlerini 
de içerecek biçimde eşit aralıklı num tane değeri oluşturarak onu bir NumPy dizisi 
olarak vermektedir. linspace ile elde edilecek eleman sayısı belli olduğu için 
arange fonksiyonu yerine genellikle programcılar bunu tercih etmektedir. linspace 
fonksiyonun dtype parametresi de vardır. Bu parametre için argüman girilmezse 
default detype np.float64 olacak biçimde belirlenir. 

import numpy  as np

a = np.linspace(0, 10, 10)
# array([ 0.        ,  1.11111111,  2.22222222,  3.33333333,  4.44444444,
        5.55555556,  6.66666667,  7.77777778,  8.88888889, 10.        ])
------------------------------------------------------------------------------------
"""

# ---------- shape ------------
"""
------------------------------------------------------------------------------------
Bir NumPy dizisinin boyutlarını (yani kaça kaçlık olduğunu) shape isimli özniteliği 
ile elde edebiliriz. shape özniteliği bize boyutları belirten bir demet vermektedir.
Örneğin:

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a.shape) # (3,4)
------------------------------------------------------------------------------------
"""

# ---------- reshape ------------
"""
------------------------------------------------------------------------------------
Bir NumPy dizisinin boyutları değiştirilebilir. Bunun için ndarray sınıfının 
reshape metodu ya da reshape fonksiyonu kullanılabilemktedir. Default durumda 
ndarray elemanları C Programlama Dilindeki gibi satırsal biçimde belleğe tek 
boyutlu olarak yerleştirilmektedir.

Örneğin aşağıdaki gibi bir NumPy dizisi bulunuyor olsun:

1 2 3 4 
5 6 7 8 

Bu aslında bu dizi bellekte şu sırada tek boyutlu biçimde tutulmaktadır:

1 2 3 4 5 6 7 8 

Şimdi biz bu diziyi 4x2 olarak reshape yaparsak dizi şu hale gelir:

1 2
3 4
5 6
7 8

Yani reshape işlemini şöyle düşünmelisiniz: Sanki önce çok boyutlu dizi tek boyuta 
dönüştürülüp, yeniden diğer boyuta dönüştürülmektedir. 

reshape işleminden yeni bir NumPy dizisi elde edilmektedir.
------------------------------------------------------------------------------------

import numpy  as np
a = np.random.randint(0, 100, (5, 4))
print(a)
print()

b = a.reshape(2,10)
print(b)
------------------------------------------------------------------------------------

Çok boyutlu dizilerin tek boyutlu hale getirilmesi çokça gereksinim duyulan bir 
işlemdir. Örneğin elimizde 5x4'lük bir NumPy dizisi olsun. Biz bunu tek boyutlu 
bir dizi haline getirmeye çalışalım. Tabii bu işlemi reshap metoduyla ya da 
fonksiyonuyla yapabiliriz. Örneğin;

a = np.random.randint(0, 100, (5, 4))
b = a.reshape(20)

Ancak bu işlem için ravel isimli bir metot ve global bir fonksiyon da 
bulundurulmuştur. ravel bize reshape işleminde olduğu gibi bir view nesnesi 
vermektedir. Örneğin:

c = a.ravel()

------------------------------------------------------------------------------------
"""

# ---------- slicing (dilimleme)------------
"""
------------------------------------------------------------------------------------
NumPy dizileri üzerinde dilimleme (slicing) yapılabilir. Dilimleme işleminde tamamen
Python listelerindeki semantik uygulanmaktadır. Dilimleme her boyut için ayrı ayrı 
yapılabilmektedir. Dilimleme işleminden yeni bir NumPy dizisi view nesnesi olarak 
elde edilmektedir. Dilimleme ilk boyuttan başlanarak boyut gerçekleştirilmektedir.

import numpy as np

a = np.random.randint(102,357, (7,7), dtype= 'int32')
print(a)
print()

print(a[1,2])
print()

print(a[2])
print()

print(a[2, 3:7])
print()

print(a[:, 2:5])
print()


a[:, 2:5] = 0
print(a)
print()
------------------------------------------------------------------------------------
"""

# bool indeksleme 
"""
------------------------------------------------------------------------------------
Bir NumPy dizisine bool indeksleme uygulanabilir. bool indeksleme için dizi uzunluğu ve boyutu kadar bool türden dolaşılabilir bir nesne
girilir. Bu dolaşılabilir nesnedeki True olan elemanlara karşı gelen dizi elemanları elde edilmektedir.

import numpy as np
a = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ])

b = [True, False, True, False, True, True, False, True, True]

c = a[b]

print(c)
------------------------------------------------------------------------------------
"""
# Matris çarpımı
"""
------------------------------------------------------------------------------------

import numpy as np

a = np.random.randint(1,23,(5,5))
b = np.random.randint(1,23,(5,5))
print(a)
print()
print(b)
print()

print(a@b)
print()

print(np.matmul(a,b)) 
------------------------------------------------------------------------------------
"""

#  ----------------------------- Pandas  -----------------------------
"""
------------------------------------------------------------------------------------
Pandas kütüphanesi NumPy kütüphanesinin üzerine kurulmuş durumdadır. Yani seviye 
bakımından NumPy'dan biraz daha yüksek seviyededir. 

Pandas'ta sütunlardan ve satırlardan oluşan veri kümeleri DataFrame isimli bir 
sınıf ile temsil edilmektedir. Veri kümesindeki belli bir sütun ise Series isimli 
sınıfla temsil edilir. DataFrame sınıfını Series nesnelerini tutan bir sınıf 
olarak düşünebiliriz. 
------------------------------------------------------------------------------------

Series nesnelerinin de tıpkı NumPy dizilerinde olduğu gibi bir dtype türü vardır. 
Bir Series nesnesi yaratılırken nesnenin dtype türü dtype parametresiyle belirtilebilir. 
Pandas içerisinde ayrı dtype sınıfları yoktur. Aslında Pandas Series bilgilerini 
NumPy dizisi olarak saklamaktadır. Dolayısıyla Series nesnesi yaratılırken dtype 
bilgisi NumPy dtype türü olarak belirtilir. Örneğin:

import pandas as pd
s = pd.Series([10, 20, 30, 40, 50], dtype=np.float32)
print(s)

!!! Series nesneleri MUTABLE nesnelerdir. Değiştirilebilir.
------------------------------------------------------------------------------------

Series nesnelerinin elemanlarına erişmek için üç yol vardır. Series nesnesi s olmak üzere:

1) Doğrudan köşeli parantez operatörü ile. Yani s[...] biçiminde.
2) loc örnek özniteliği ve köşeli parantez operatörü ile. Yani s.loc[...] biçiminde
3) iloc örnek özniteliği ve köşeli parantez operatörü ile. Yani s.iloc[...] biçiminde

Seeries nesneleri "değiştirilebilir (mutable)" nesnelerdir. Bir Series nesnesine 
erişip onu değiştirebiliriz. 

Series nesnesinin index ile belirtilen (ya da index belirten değerlerine) 
"etiket (label)" da denilmektedir. Örneğin:


------------------------------------------------------------------------------------
s.iloc[...] biçimindeki erişimde ise köşeli parantez içerisine her zaman sıra 
numarası yerleştirilmek zorundadır. Biz burada köşeli parantez içerisine etiket 
yerleştiremeyiz.
    
Yani loc örnek özniteliği ile erişimde köşeli parantez içerisinde her zaman etiket 
bulundurulması gerekmektedir. Bu bakımdan s[...] erişimi ile s.loc[...] erişimi
birbirine benzemektedir fakat aralarında farklılık vardır.     
------------------------------------------------------------------------------------

Doğrudan indekslemede hem etiket hem de sıra numarası bir arada kullanılamz. 
Ancak bunlardan biri kullanılabilir. Örneğin:

import pandas as pd

s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'], dtype='float32')
k = s[['b', 'a', 'e']]
print(k)
print()

k= s.iloc[[1, 3, 2]]
print(k)
------------------------------------------------------------------------------------

Series nesnesinin içerisindeki değerler values isimli örnek özniteliği ile bir 
NumPy dizisi olarak elde edilebilmektedir. Aslında Series nesnesi zaten değerleri 
NumPy dizisi içerisinde tutmaktadır. values elemanı da bize doğrudan aslında bu 
diziyi verir. Bu dizide değişiklik yaptığımızda Series nesnesinin elemanında 
değişiklik yapmış oluruz. Örneğin:

s = pd.Series([10, 20, 30, 40, 50], dtype='float32')
a = s.values
print(a)

Series sınıfının to_numpy metodu values örnek özniteliği gibidir. Ancak to_numpy 
değişik seçeneklere de sahiptir. Default durumda to_numpy metodu ile values örnek 
özniteliği aynı Numpy dizisini vermektedir. Fakat örneğin to_numpy metodunda 
copy=True geçilirse metot bize kopyalama yaparak başka bir Numpy dizisi verir. 

s = pd.Series([10, 20, 30, 40, 50], dtype='float32')

a = s.values
print(id(a))
print(a)

a = s.to_numpy()
print(id(a))
print(a)

print('-----')

a = s.to_numpy(copy=True)
print(a)
print(id(a))

Özetle bir Series nesnesi içerisindeki değerleri NumPy dizisi olarak almak istersek 
values örnek özniteliğini ya da to_numpy metodunu kullanabiliriz.

a = s.to_numpy()
print(a)


Eğer Series nesnesi içerisindeki değerleri bir Python listesi biçiminde elde etmek 
istersek Series sınıfının to_list metodunu kullanabiliriz. 

Tabii to_list her çağrıldığında aslında bize farklı bir list nesnesi verecektir.
------------------------------------------------------------------------------------

Series nesnesinden eleman silmek için Series sınıfının drop metodu kullanılmaktadır 
(drop isimli bir fonksiyon yoktur). Bu metot her zaman etiket temelinde çalışır. 
Hiçbir zaman sıra numarasıyla çalışmaz. Biz tek bir etiket de kullanabiliriz. Bir 
grup etiketi dolaşılabilir bir nesne biçiminde de metoda verebiliriz. Metot default 
durumda "inplace" silme işlemi yapmaz. Bize silinmiş yeni bir Series nesnesi verir. 
Örneğin:
        
s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'], dtype='float32')
print(s)
result = s.drop(['a', 'e'])
print()
print(result)


Metodun inplace parametresi True geçilirse silme işlemi nesne üzerinde yapılır. 
Bu durumda metot None değerine geri döner.

s.drop(['a', 'e'], inplace=True)
print(s)            
------------------------------------------------------------------------------------

Series sınıfının pek çok faydalı metodu vardır. Bu metotlar bize yaptıklar işlem 
sonucunda yeni bir Series nesnesi verirler. Aslında bu metotlar NumPy metotlarına 
çok benzemektedir. NumPy'da pek çok işlem hem metotlarla hem de fonksiyonlarla 
yapılabilmektedir. Ancak Pandas'ta ağırlıklı olarak metotlar bulundurulmuştur. 
Yani pek çok fonksiyon yalnızca metot biçiminde bulundurulmuştur.

abs metodu elemanların mutlak değerlerini elde eder. add metodu karşılıklı elemanları 
toplar (yani + operatörü ile yapılanı yapar). argmax, argmin, argsort metotoları 
sırasıyla en büyük elemanın indeksisni, en küçük elemanın indeksini ve sort 
edilme durumundaki indeksleri vermektedir.

s = pd.Series([12, 8, -4, 2, 9], dtype='float32')
print(s.abs())
print(s.argmin())
print(s.argmax())
print(s.argsort()) 
------------------------------------------------------------------------------------

dropna metodu eksik verileri atmak için kullanılmaktadır. Yani NaN değerleri 
Series nesnesinden silinir.

s = pd.Series([3, None, 7, 9, None, 10], dtype='float32')
print(s)
print()
result = s.dropna()
print(result)
print()

fillna isimli metot eksik verileri (yani NaN olan elemanları) spesifik bir değerle 
doldurmaktadır. Örneğin biz eksik verileri aşağıdaki gibi ortalamayla doldurabiliriz:

reul= s.fillna(s.mean())
print(reul)        
------------------------------------------------------------------------------------
"""

# DataFrame
"""
------------------------------------------------------------------------------------
Pandas'taki en önemli veri yapısı DataFrame denilen veri yapısıdır. DataFrame tipik 
olarak istatistiksel veri kümesini temsil etmek için düşünülmüştür. DataFrame 
nesnesinin sütunlardan oluşan matrisel bir yapısı vardır. Aslında DataFrame nesnesi 
Series nesnelerinden oluşmaktadır. Yani DataFrame nesnelerinin sütunları Series 
nesneleridir. 

NumPy dizilerinin elemanları aynı türden olur. Her ne kadar elemanları aynı türden 
olmayan NumPy dizileri de oluşturulabiliyorsa da (örneğin dtype='object' diyerek) 
bu biçimde NumPy dizilerinin uygulamada kullanımı yoktur. O halde Pandas kütüphanesi 
aslında sütunları farklı türlerden olabilen DataFrame denilen bir veri yapısı 
sunmaktadır. İstatistik ve veri bilimindeki "veri kümeleri (datasets)" ham durumda 
böyle bir yapıya sahiptir. Bu bakımdan veri kümeleri veritabanlarındaki tablolara 
da benzetilebilir.

DataFrame nesnesi iki boyutlu bir Python listesi ile oluşturulabilir. Eğer index 
parametresi ve columns parametresi belirtilmezse oluşturulan DataFrame nesnesinin 
satır etiketleri ve sütun etiketleri 0, 1, 2, ... biçiminde atanır
------------------------------------------------------------------------------------
import pandas as pd
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'b', 'c'], columns=['x', 'y', 'z'])
print(df)
------------------------------------------------------------------------------------

Bir DataFrame nesnesi iki boyutlu bir NumPy dizisi ile de yaratılabilir. Örneğin:

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
df = pd.DataFrame(a, columns=['a', 'b', 'c'], index= [3,7,4])
print(df)
print()

print(df.loc[7]) # df.loc[0] -> indexError
print()

print(df.iloc[1]) # df.iloc[7] -> indexError
------------------------------------------------------------------------------------

DataFrame nesnesi bir sözlük ile de yaratılabilir. Bu durumda sözlüğün anahtarları 
sütun isimlerini, değerleri de sütunlardaki değerleri belirtir. Örneğin:
    
d = {'Adı Soyadı': ['Kaan Aslan', 'Ali Serçe', 'Ayşe Er'], 'Boy': [182, 174, 168], 'Kilo': [78, 69, 56]}
df = pd.DataFrame(d)
print(df)
------------------------------------------------------------------------------------

DataFrame üzerinde bir sütun insert etmek için DataFrame sınıfının insert metodu 
metodu kullanılabilmektedir. insert metodunun birinci parametresi her zaman insert 
edilecek sütunun indeks numarasını alır. İkinci parametre indeks edilecek sütunun 
ismini (ayni etiketini), üçüncü parametre ise sütun bilgilerini almaktadır. insert 
metodu "in-place" insert işlemi yapmaktadır. Yani DatFrame nesnesinin kendi 
üzerinde ekleme yapılmaktadır.

import pandas as pd
d = {'Adı': ['Ali', 'Veli', 'Selami', 'Ayşe', 'Fatma'], 'Kilo': [48.3, 56.7, 92.3, 65.3, 72.3], 'Boy': [172, 156, 182, 153, 171]}
df = pd.DataFrame(d)
print(df)
print('----------')

bmi = df['Kilo'] / (df['Boy'] / 100) ** 2
print(bmi)
print(type(bmi))
print('----------')

df.insert(3, 'Vücut Kitle Endeksi', bmi)
print(df)
------------------------------------------------------------------------------------
"""

#  ----------------------------- Statistics  -----------------------------
"""
------------------------------------------------------------------------------------
İstatistikte verilerin merkezine ilişkin bilgi veren ölçülere "merkezi eğilim ölçüleri 
(measures of central tendency)" denilmektedir. Merkezi eğilim ölçülerinin en yaygın 
kullanılanı "aritmetik ortalamadır". Aritmetik ortalama (mean) değerlerin toplanarak 
değer sayısına bölünmesiyle elde edilmektedir. 

Aritmetik ortalama hesaplamak için çeşitli kütüphanelerde çeşitli fonksiyonlar hazır 
olarak bulunmaktadır. Örneğin Python'ın standart kütüphanesindeki statistics modülünde 
bulunan mean fonksiyonu aritmetik ortalama hesaplamaktadır. 

import statistics
a = [1, 2, 7, 8, 1, 5]
statistics.mean(a)

axis=0  -> satır
axis=1  -> sutun

mean fonksiyonu herhangi bir dolaşılabilir nesneyi parametre olarak alabilmektedir. 

NumPy kütüphanesindeki mean fonksiyonu axis temelinde (yani satırsal ve sütunsal biçimde) 
ortalama hesaplayabilmektedir. Örneğin:

import numpy as np
a = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
np.mean(a, axis=0)
# array([4.66666667, 5.66666667, 6.66666667])


Pandas kütüphanesinde Series ve DataFrame sınıflarının mean metotları aritmetik 
ortalama hesabı yapmaktadır. DataFrame sınıfının mean metodunda default axis 0 
biçimindedir. Yani sütunsal ortalamalar elde edilmektedir. Örneğin:

import pandas as pd
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(df)
print()
print(df.mean())
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Diğer bir merkezi eğilim ölçüsü de "medyan (median)" denilen ölçüdür. Medyan sayıların 
küçükten büyüğe sıraya dizildiğinde ortadaki değerdir. Ancak sayılar çift sayıda 
ise sayıların tam ortasında bir değer olmadığı için ortadaki iki değerin aritmetik 
ortalaması medyan olarak alınmaktadır. Medyan işlemi uç değerlerden (outliers) 
etkilenmez. Ancak medyan işlemi aritmetik ortalamadan daha fazla zaman alan bir 
işlemdir. Çünkü medyan için önce değerlerini sıraya dizilmesi gerekmektedir. 
Dolayısıyla  medyan işlemi O(N log N) karmaşıklıta bir işlemdir.

import statistics
a = [1, 23, 56, 12, 45, 21]
statistics.median(a)

NumPy kütüphanesinde medyan işlemi eksensel biçimde median fonksiyonuyla yapılabilmektedir.

import numpy as np
a = np.random.randint(1, 100, (10, 10))

np.median(a, axis=0)
# array([33.5, 56.5, 69. , 48. , 57.5, 39. , 54.5, 32. , 61.5, 40.5])


df = pd.DataFrame(a)
df.median()
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Merkezi eğilim ölçülerinin bir diğeri de "mod" denilen ölçüdür. Bir grup verideki 
en çok yinelenen değere "mod (mode)" denilmektedir. Mod özellikle kategorik ve 
sıralı ölçeklerde ortalamanın yerini tutan bir işlem olarak kullanılmaktadır. 
Mod işlemi genel olarak O(N log N) karmaşıklıkta yapılabilmektedir. (Tipik mod 
algoritmasında değerler önce sıraya dzilir. Sonra yan yana aynı değerlerden kaç 
tane olduğu tespit edilir.)

import statistics
a = [1, 3, 3, 4, 2, 2, 5, 2, 7, 9, 5, 3, 5, 7, 5]
statistics.mode(a)

NumPy kütüphanesinde mod işlemini yapan bir fonksiyon bulunmamaktadır. Ancak SciPy 
kütüphanesinde mod işlemi için stats modülü içerisindeki mode fonksiyonu kullanılabilir. 
Bu fonksiyon yine eksensel işlemler yapabilmektedir. mode fonksiyonu ModeResult 
isimli bir sınıf türünden tuple sınıfından türetilen bir sınıf türünden bir nesne 
verir. Bu sınııfın mode ve count örnek öznitelikleri en çok yinelenen değerleri ve 
onların sayılarını bize vermektedir. ModeResult sınıfı bir çeşit demet özelliği 
gösterdiği için demet gibi de kullanılabilir. 

import scipy.stats
import numpy as np

a = np.random.randint(1, 10, (20, 10))
mr = scipy.stats.mode(a, axis=0)
print(type(mr))  # <class 'scipy.stats._stats_py.ModeResult'>

mr.mode
#  array([[4, 1, 2, 3, 5, 8, 4, 6, 8, 1]])

mr.count
# array([[4, 4, 6, 4, 4, 4, 4, 4, 5, 4]])



Pandas kütüphanesinde Series ve DataFrame sınıflarının mode metotları da mode 
işlemi yapmaktadır.

import pandas as pd
a = np.random.randint(1, 10, (2, 10))
df = pd.DataFrame(a)
df.mode()

DataFrame sınıfının mode metodu bize bir DataFrame nesnesi vermektedir. Uygulamacı 
genellikle bunun ilk satırı ile ilgilenir. Diğer satırlar eşit miktarda tekrarlanan 
elemanlardan oluşmaktadır. Tabii belli bir sütunda eşit miktarda tekrarlanan elemanların
sayısı az ise artık geri döndürülen DataFrame'in o sütuna ilişkin satırlarında 
NaN değeri bulunacaktır.
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

Değerlerin merkezine ilişkin bilgiler dağılım hakkında iyi bir fikir vermeyebilir. 
Örneğin iki ülkede kişi başına düşen ortalama yıllık gelir (gayri safi milli hasıla) 
15000 dolar olabilir. Ancak bu iki ülke arasında gelir dağılımında önemli farklılıklar 
bulunuyor olabilir. O halde değerlerin ortalamasının yanı sıra onların ortalamaya 
göre nasıl yayıldıkları da önemlidir. İstatistikte değerlerin ortalamaya göre 
yayılımı için "merkezi yayılım ölçüleri (measures of dispersion)" denilen bazı 
ölçüler kullanılmaktadır. Merkezi yayılım ölçüleri aslında değerlerin ortalamadan 
ortalama uzaklığını belirlemeyi hedeflemektedir

Merkezi yayılım ölçüsü olarak "değerlerin ortalamadan ortalama mutlak uzaklığına" 
başvurulabilir. Burada mutlak değer alınmasının nedeni uzaklıkları negatif olmaktan 
kurmak içinidir. Bu durumda ortalama 0 çıkmaz. Ancak bu yöntem de aslında çok iyi 
bir yöntem değildir. Aşağıda aynı dilimin ortalamadan ortalama mutlak uzaklığı 
hesaplanmışır.

   
import numpy as np

a = np.array([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
mean = np.mean(a)
print(mean)                # 5

result = np.mean(np.abs(a - mean))
print(result)               # 2.5
------------------------------------------------------------------------------------

Aslında ortalamadan ortalama uzaklık için "standart sapma (standard deviation)" 
denilen ölçü tercih edilmektedir. Standart sapmada ortalamadan uzaklıkların mutlak 
değeri değil kareleri alınarak negatiflikten kurtulunmaktadır. Kare alma işlemi 
değerleri daha fazla farklılaştırmaktadır. Yani aynı değerlerin oluşma olasılığı 
kare alma sayesinde daha azalmaktadır. Aynı zamanda bu işlem bazı durumlarda başka 
faydalara da yol açmaktadır. (Örneğin ileride bu kare alma işlemlerinin optimizasyon
problemlerinde uygun bir işlem olduğunu göreceğiz.)

import numpy as np

def sd(a, ddof = 0):
    return np.sqrt(np.sum((a - np.mean(a)) ** 2)  / (len(a)  - ddof))

a = [1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8]
result = sd(a)
print(result)           # 2.7688746209726918
------------------------------------------------------------------------------------

Python'ın Standart Kütüphanesinde statistics modülü içerisinde standart sapma hesabı 
yapan stdev ve pstdev fonksiyonları bulunmaktadır. stdev fonksiyonu (n - 1)'e bölme 
yaparken, pstdev  (buradaki 'p' harfi "population" sözcüğünden gelmektedir) fonksiyonu 
n'e bölme yapmaktadır. 

import statistics

a = [1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8]

std = statistics.stdev(a)
print(std)                          # 2.891995221924885  

std = statistics.pstdev(a)
------------------------------------------------------------------------------------

NumPy kütüphanesinde std isimli fonksiyon eksensel standart sapma hesaplayabilmektedir. 
Fonksiyonun ddof parametresi default durumda 0'dır. Yani default durumda fonksiyon 
n'e bölme yapmaktadır.

import numpy as np

a = np.array([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
result = np.std(a)
print(result)                       # 2.7688746209726918 
------------------------------------------------------------------------------------

Pandas kütüphanesinde de Series ve DataFrame sınıflarının std isimli metotları
eksensel standart sapma hesabı yapabilmektedir. Ancak bu metotlarda ddof parametresi 
default 1 durumundadır. Yani bu metotlar default durumda (n - 1)'e bölme yapmaktadır.

import pandas as pd

s = pd.Series([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
result = s.std()
print(result)                       # 2.891995221924885        

s = pd.Series([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
result = s.std(ddof=0)
print(result)                       # 2.7688746209726918 
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

Standart sapmanın karesine "varyans (variance)" denilmektedir. Varyans işlemi standart 
kütüphanedeki statistics modülünde bulunan variance ve pvariance fonksiyonlarıyla 
yapılmaktadır. NumPy kütüphanesinde varyans işlemi var fonksiyonuyla ya da ndarray 
sınıfının var metoduyla, Pandas kütüphenesinin Series ve DataFrame sınıflarındaki 
var metoduyla yapılmaktadır. Yine NumPy'daki variance fonksiyonundaki ddof default 
olarak 0, Pandas'taki ddof ise 1'dir.

Pekiyi neden standart sapma varken ayrıca onun karesi için varyans terimi uydurulmuştur? 
İşte istatistikte pek çok durumda aslında doğrudan ortalamadan farkların karesel 
ortalamaları (yani standart sapmanın karesi) kullanılmaktadır. Bu nedenle bu hesaba 
ayrı bir isim verilerek anlatımlar kolaylaştırılmıştır.


import statistics
import numpy as np
import pandas as pd

data = [1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8]

result = statistics.pvariance(data)
print(result)               # 7.666666666666667

result = np.var(data)
print(result)               # 7.666666666666667

s = pd.Series(data)
result = s.var(ddof=0)
print(result)               # 7.666666666666667
------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------
Bir deney sonucunda oluşacak durum baştan tam olarak belirlenemiyorsa böyle deneylere 
"rassal deney (random experiment)" denilmektedir. Örneğin bir paranın atılması 
deneyinde para "yazı" ya da "tura" gelebilir. O halde "paranın atılması" rassal 
bir deneydir. Benzer biçimde bir zarın atılması, bir at yarışında hangi atın birinci 
olacağı gibi deneyler rassal deneylerdir. 

Bir deneyin sonucu öndecen bilinebiliyorsa bu tür deneylere "deterministik deneyler" 
de denilmektedir. 

Bir rassal deney sonucunda oluşabilecek tüm olası durumların kümesine "örnek uzayı 
(sample space)" denilmektedir.

Olasılığın (probablity) değişik tanımları yapılabilmektedir. Olasılığın en yaygın 
tanımlarından birisi ------>>  "göreli sıklık (relative frequency)" tanımıdır. 

Bu tanıma göre bir rassal olay çok sayıda yinelendikçe elde edilen olasılık değeri 
belli bir değere yakınsamaya başlar. Örneğin bir paranın 100 kere atılmasında 
50 kere yazı 50 tura gelmeyebilir. 


Ancak para sonsuz sayıda atılırsa (ya da çok fazla sayıda atılırsa) tura gelme 
sayısının paranın atılma sayısına oranı 0.5'e yakınsayacaktır. 

Buna istatistike ---->> "büyük sayılar yasası (law of large numbers)" da denilmektedir. 

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Olasılıkta ve istatistikte en çok kullanılan temel kavramlardan biri "rassal değişken 
(random variable)" denilen kavramdır. Her ne kadar "rassal değişken" isminde bir 
"değişken" geçiyorsa da aslında rassal değişken bir fonksiyon belirtmektedir. 
Rassal değişken bir rassal deney ile ilgilidir. Bir rassal deneyde örnek uzayın 
her bir elemanını (yani basit olayını) reel bir değere eşleyen bir fonksiyon 
belirtmektedir. Rassal değişkenler genellikle "sözel biçimde" ifade edilirler. 
Ancak bir fonksiyon belirtirler. Rassal değişkenler matematiksel gösterimlerde 
genellikle büyük harflerle belirtilmektedir. Örneğin:

- R rassal değişkeni "iki zar atıldığında zarların üzerindeki sayıların toplamını"
belirtiyor olsun. Burada aslında R bir fonksiyondur. Örnek uzayın her bir elemanını 
bir değere eşlemektedir. Matematiksel gösterimle R rassal değişkeni şöyle belirtilebilir:

R: S -> R

Burada R'nin  örnek uzayından R'ye bir fonksiyon belirttiği anlaşılmalıdır. Burada 
R fonksiyonu aşağıdaki gibi eşleme yapmaktadır:

(1, 1) -> 2
(1, 2) -> 3
(1, 3) -> 4
...
(6, 5) -> 11
(6, 6) -> 12

K rassal değişkeni "rastgele seçilen bir kişinin kilosunu belirtiyor" olsun. Bu 
durumda örnek uzayı aslında dünyaki tüm insanlardır. Burada K fonksiyonu da her 
insanı onun kilosuna eşleyen bir fonksiyondur. 

Rassal değişkenler kümeler üzerinde işlemler yapmak yerine gerçek sayılar üzerinde 
işlem yapmamızı sağlayan, anlatımlarda ve gösterimlerde kolaylık sağlayan bir kavramdır. 


Rassal değişkenler tıpkı matematiksel diğer fonksiyonlarda olduğu gibi "kesikli 
(discrete)" ya da "sürekli (continuous)" olabilmektedir. eğer bir rassal değişken 
(yani fonksiyon) teorik olarak belli bir aralıkta tüm gerçek sayı değerlerini 
alabiliyorsa böyle rassal değişkenlere "sürekli (continous)" rassal değişkenler 
denilmektedir. Ancak bir rassal değişken belli bir aralıkta yalnızca belli gerçek 
sayı değerlerini alabiliyorsa bu rassal değişkenlere "kesikli (discrete)" rassal
değişkenler denilmektedir.

Örneğin "iki zarın atılmasında üste gelen sayılar toplamını belirten R rassal 
değişkeni" kesiklidir. Çünkü yalnızca belli değerleri alabilmektedir. Ancak 
"rastgele seçilen bir kişinin kilosunu belirten" K rassal değişkeni süreklidir. 
Çünkü teorik olarak belli bir aralıkta tüm gerçek değerleri alabilir.(Biz kişilerin 
kilolarını yuvarlayarak ifade etmekteyiz. Ancak aslında onların kiloları belli 
aralıktaki tüm gerçek değerlerden biri olabilir.)

------------------------------------------------------------------------------------
Yapay zeka ve makine öğrenmesinde sürekli rassal değişkenler daha fazla karşımıza 
çıkmaktadır. Bu nedenle biz sürekli rassal değişkenler ve onların olasılıkları 
üzerinde biraz daha duracağız. 

Sürekli bir rassal değişkenin aralıksal olasılıklarını hesaplama aslında bir 
"intergral" hesabı akla getirmektedir. İşte sürekli rassal değişkenlrin aralıksal 
olasılıklarının hesaplanması için kullanılan fonksiyonlara -----> 

"olasılık yoğunluk fonksiyonları (probability density functions)" denilmektedir. 

Birisi bize bir rassal değişkenin belli bir aralıktaki olasılığını soruyorsa o 
kişiin bize o rassal değişkene ilişkin "olasılık yoğunluk fonksiyonunu" vermiş 
olması gerekir. Biz de örneğin P{x0 < X < x1} olasılığını x0'dan x1'e f(x)'in 
integrali ile elde ederiz. 

Bir fonksiyonun olasılık yoğunluk fonksiyonu olabilmesi için -sonsuzdan + sonsuza 
integralinin (yani tüm eğri altında kalan alanın) 1 olması gerekir. Bir rassal 
değişkenin olasılık yoğunluk fonksiyonuna "o rassal değişkenin dağılımı" da 
denilmektedir.

------------------------------------------------------------------------------------
	
Değişik ortalama ve standart sapmaya ilişkin sonsuz sayıda Gauss eğrisi çizilebilir. 
Ortalaması 0, standart sapması 1 olan normal dağılıma "standart normal dağılım" da 
denilmektedir. Genellikle istatistiktre standart normal dağılımdaki X değerlerine
"Z değerleri" denilmektedir. Aşağıdaki örnekte Gauss eğrisi çizdirilmiştir.

import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu = 0, std = 1):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.e ** (-0.5 * ((x - mu) / std) ** 2)

# yukarıdaki formül -->  'probability denstiy function'    

x = np.linspace(-5, 5, 1000)
y = gauss(x)

plt.title('Gauss Function')
plt.plot(x, y)
plt.show()

------------------------------------------------------------------------------------
Yukarıdaki çizimde eksenleri kartezyen koordinat sistemindeki gibi de gösterbiliriz. 
 
import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu = 0, std = 1):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.e ** (-0.5 * ((x - mu) / std) ** 2)
  
    
def draw_gauss(mu = 0, std = 1):
    x = np.linspace(-5 * std + mu, 5 * std + mu, 1000)
    y = gauss(x, mu, std)
    
    mu_y = gauss(mu, mu, std)
    
    plt.figure(figsize=(10, 4))
    plt.title('Gauss Function', pad=10, fontweight='bold')
    axis = plt.gca()
    
    axis.set_ylim([-mu_y * 1.1, mu_y * 1.1])
    axis.set_xlim([-5 * std + mu, 5 * std + mu])
    axis.set_xticks(np.arange(-4 * std + mu, 5 * std + mu, std))
    # axis.set_yticks(np.round(np.arange(-mu_y, mu_y, mu_y / 10), 2))
    axis.spines['left'].set_position('center')
    axis.spines['top'].set_color(None)
    axis.spines['bottom'].set_position('center')
    axis.spines['right'].set_color(None)
    axis.plot(x, y)
    plt.show()

draw_gauss(100, 15)

------------------------------------------------------------------------------------
Normal dağılımda eğri altında kalan toplam alanın 1 olduğunu belirtmiştik. Bu dağılımda 
toplaşmanın ortalama civarında olduğu eğirinin şeklinden anlaşılmaktadır. Gerçektende 
normal dağılımda ortalamadan bir standart sapma soldan ve sağdan kaplanan alan 
yani P{mu - std < X < mu + std} olasılığı 0.6827, ortalamadna iki standart sapma 
soldan ve sağdan kaplanan alan yani P{mu - std * 2< X < mu + std * 2} olasılığı 
0.9545 biçimindedir. 

Matplotlib'te bir eğrinin altındaki aalanı boyamak için fill_between isimli fonksiyon 
kullanılmaktadır. Bu fonksiyon axis sınıfının bir metodu olarak da bulundurulmuştur. 
Aşağıdaki örnekte eğrinin altındaki belli bir alan fill_between metodu ile boyanmıştır.

import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu = 0, std = 1):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.e ** (-0.5 * ((x - mu) / std) ** 2)
      
def draw_gauss(mu = 0, std = 1, fstart= 0, fstop = 0):
    x = np.linspace(-5 * std + mu, 5 * std + mu, 1000)
    y = gauss(x, mu, std)
    
    mu_y = gauss(mu, mu, std)
    
    plt.figure(figsize=(10, 4))
    plt.title('Gauss Function', pad=10, fontweight='bold')
    axis = plt.gca()
    
    axis.set_ylim([-mu_y * 1.1, mu_y * 1.1])
    axis.set_xlim([-5 * std + mu, 5 * std + mu])
    axis.set_xticks(np.arange(-4 * std + mu, 5 * std + mu, std))
   # axis.set_yticks(np.round(np.arange(-mu_y, mu_y, mu_y / 10), 2))
    axis.spines['left'].set_position('center')
    axis.spines['top'].set_color(None)
    axis.spines['bottom'].set_position('center')
    axis.spines['right'].set_color(None)
    axis.plot(x, y)
    
    x = np.linspace(fstart, fstop, 1000)
    y = gauss(x, mu, std)
    axis.fill_between(x, y)
    plt.show()

draw_gauss(100, 15, 85, 115)

------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
Kümülatif dağılım fonksiyonu (cummulative distribution function) belli bir değere 
kadar tüm birikimli olasılıkları veren fonksiyondur. Genellikle F harfi gösterilmektedir. 
Örneğin F(x0) aslında P{X < x0} anlamına gelmektedir. Normal dağılımda F(x0) değeri 
aslında eğride X değerinin x0 olduğu noktadan soldaki tüm eğri altında kalan alanı 
belirtmektedir. (Başka bir deyişle sürekli dağılımlarda F(x0) değeri "-sonsuzdan 
x0'a kadar olasılık yoğunluk fonksiyonunun integraline eşittir)
------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
Normal dağılımla ilgili işlemleri yapabilmek için Python standart kütüphanesinde 
statistics modülü içerisinde NormalDist isimli bir sınıf bulundurulmuştur. Programcı 
bu sınıf türünden bir nesne yaratır. İşlemlerini bu sınıfın metotlarıyla yapar.
NormalDist nesnesi yaratılırken __init__ metodu için ortalama ve standart sapma 
değerleri girilir. (Bu değerler girilmezse ortalama için sıfır, standart sapma 
için 1 default değerleri kullanılmaktadır.)

------------------------------------------------------------------------------------
NormalDist sınıfının cdf (cummulative distribution function) isimli metodu verilen 
x değeri için eğrinin solunda kalan toplam alanı yani kümülatif olasılığı bize 
vermektedir. Örneğin standart normal dağılımda x = 0'ın solunda alan 0.5'tir.

import statistics

nd = statistics.NormalDist()

result = nd.cdf(0)
print(result)           # 0.5

------------------------------------------------------------------------------------
Örneğin biz ortalaması 100, standart sapması 15 olan bir normal dağılımda 
P{130 < x < 140} olasılığını aşağıdaki gibi elde edebiliriz:

nd = statistics.NormalDist(100, 15)
result = nd.cdf(140) - nd.cdf(130)
print(result)   # 0.018919751380589434

Şöyle bir soru sorulduğunu düşünelim: "İnsanların zekaları ortalaması 100, standart 
sapması 15 olan normal dağılıma uygundur. Bu durumda zeka puanı 140'ın yukarısında 
olanların toplumdaki yüzdesi nedir?". Bu soruda istenen şey aslında normal dağılımdaki
P{X > 140} olasılığıdır. Yani x ekseninde belli bir noktanın sağındaki kümülatif 
alan sorulmaktadır. Bu alanı veren doğrudan bir fonksiyon olmadığı için bu işlem 
1 - F(140) biçiminde ele alınarak sonuç elde edilebilir. Yani örneğin:

nd = statistics.NormalDist(100, 15)
result = 1 - nd.cdf(140)

------------------------------------------------------------------------------------
Belli bir kümülatif olasılık değeri için x değerinin bulunması işlemi de NormalDist 
sınıfının inv_cdf metoduyla yapılmaktadır. Örneğin standart normal dağılımda 0.99 
olan kümülatif olasılığın Z değeri aşağıdaki gibi bulunabilir:

nd = statistics.NormalDist()
result = nd.inv_cdf(0.99)
print(result)                       # 2.3263478740408408

result = nd.cdf(2.3263478740408408) # 0.99
print(result)
------------------------------------------------------------------------------------

Belli bir x değeri için Gauss fonksiyonunda ona karşı gelen y değeri sınıfın pdf 
metduyla elde edilmektedir. Örneğin x = 0 için standart normal dağılımda Gauss 
fonksiyonu değerini aşağıdaki gibi elde edebiliriz:

nd = statistics.NormalDist()
result = nd.pdf(0)
print(result)   # 0.3989422804014327
------------------------------------------------------------------------------------

!!!! Normal dağılmış !!!! rastgele n tane sayı üretmek için NormalDist sınıfının 
samples isimli metodu kullanılmaktadır. Bu metot bize bir liste olarak n tane 
float değer verir. Örneğin:

nd = statistics.NormalDist()
result = nd.samples(10)
print(result) 

Bu işlemden biz normal dağılmış 10 tane rasgele değerden oluşan bir liste elde ederiz. 

Biz normal dağılmış rastgele sayılardan histogram çizersek histogramımızın 
Gauss eğrisine benzemesi gerekir.


import statistics

nd = statistics.NormalDist()

result = nd.samples(10000)

import matplotlib.pyplot as plt

plt.hist(result, bins=30)
plt.show()
------------------------------------------------------------------------------------

Python'ın statistics modülündeki NormalDist sınıfı vektörel işlemler yapamamaktadır. 
Maalesef NumPy ve Pandas kütüphanelerinde normal dağılım üzerinde vektörel işlem 
yapan öğeler yoktur. Ancak SciPy kütüphanesi içerisinde pek çok dağılım üzerinde 
vektörel işlemler yapan sınıflar bulunmaktadır. Bu nedenle pratikte Python kütüphanesi 
yerine bu tür işlemler için SciPy kütüphanesi tercih edilmektedir. 
------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
scipy.stats modülü içerisindeki norm isimli singleton nesne normal dağılım üzerinde 
vektörel işlem yapan metotlara sahiptir. norm bir sınıf nesnesidir ve zaten yaratılmış 
bir biçimde bulunmaktadır. Dolayısıyla programcı doğrudan bu nesne ile ilgili sınıfın 
metotlarını çağırabilir. Genellikle programcılar bu tür nesneleri kullanmak için 
from import deyimini tercih ederler:

from scipy.stats import norm
 
norm nesnesine ilişkin sınıfın cdf isimli metodu üç parametre almaktadır:

cdf(x, loc=0, scale=1)

Buradaki x bir NumPy dizisi ya da Python dolaşılabilir nesnesi olabilir. Bu durumda 
tüm x değerlerinin kümülatif olasılıkları hesaplanıp bir NumPy dizisi olarak 
verilmektedir. Burada loc ortalamayı, scale ise standart sapmayı belirtmektedir. 
Örneğin:

from scipy.stats import norm
result = norm.cdf([100, 130, 140], 100, 15)
print(result)

------------------------------------------------------------------------------------

norm nesnesinin ilişkin olduğu sınıfın ppf (percentage point function) isimli metodu 
cdf işleminin tersini yapmaktadır. Yani kümülatif olasılığı bilindiği durumda bize 
bu kümalatif olasılığa karşı gelen x değerini verir. (Yani ppf NormalDist sınıfındaki 
inv_cdf metoduna karşılık gelmektedir.):

ppf(q, loc=0, scale=1)

from scipy.stats import norm

result = norm.ppf([0.50, 0.68, 0.95], 100, 15)
print(result)

ppf (percentage point function) ismi size biraz tuhaf gelebilir. Bu isim birikimli 
dağılım fonksiyonunun tersini belirtmek için kullanılmaktadır. ppf aslında 
"medyan (median)" kavramının genel biçimidir. Anımsanacağı gibi medyan ortadan ikiye 
bölen noktayı belirtiyordu. Örneğin standart normal dağılımda medyan 0'dır. 
Yani ortalamaya eşittir. 

İstatistikte tam ortadan bölen değil de diğer noktalardan bölen değerler için 
"percentage point" de denilmektedir. Örneğin normal dağılımda 1/4 noktasından bölen 
değer aslında birikimli dağılm fonksiyonunun 0.25 için değeridir. 

------------------------------------------------------------------------------------
norm nesnesinin ilişkin olduğu sınıfın pdf (probability density function) isimli 
metodu yine x değerlerinin Gaus eğrisindeki Y değerlerini vermektedir. Metodun 
parametrik yapısı şöyledir:

pdf(x, loc=0, scale=1) 

------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(40, 160, 1000)
y = norm.pdf(x, 100, 15)

plt.plot(x, y)

x = np.full(200, 100)       # 200 tane 100'lerden oluşan dizi
yend = norm.pdf(100, 100, 15)
y = np.linspace(0, yend, 200)
plt.plot(x, y, linestyle='--')

plt.show()

------------------------------------------------------------------------------------
norm nesnesinin ilişkin olduğu sınıfın rvs metodu ise normal dağılıma ilişkin rassal 
sayı üretmek için kullanılmaktadır. Metodun parametrik yapısı şöyledir:

    rvs(loc=0, scale=1, size=1)

statistics.NormalDist()'in sample metodunun bezeri
------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = norm.rvs(100, 15, 10000)

plt.hist(x, bins=20)
plt.show()

------------------------------------------------------------------------------------
Normal dağılımda ortalamadan birer standart sapma arasındaki bölgenin olasılığı, 
yani P{mu - sigma < x < mu + sigma} olasılığı 0.68 civarındadır.


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

result = norm.cdf(1) - norm.cdf(-1)
print(result)

x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

plt.title('Ortalamadan 1 Standart Sapma Arası Bölge', fontweight='bold')
axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_xticks(range(-4, 5))
axis.text(2, 0.3, f'{result:.3f}', fontsize=14, fontweight='bold')

plt.plot(x, y)

x = np.linspace(-1, 1, 1000)
y = norm.pdf(x)
plt.fill_between(x, y)
axis.arrow(2.5, 0.25, -2, -0.1, width=0.0255)

plt.show()

------------------------------------------------------------------------------------
Normal dağılımda ortalamadan iki standart sapma arasındaki bölgenin olasılığı, yani 
P{mu - 2 * sigma < x < mu + 2 * sigma} olasılığı 0.95 civarındadır.


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

result = norm.cdf(2) - norm.cdf(-2)
print(result)

x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

plt.title('Ortalamadan 2 Standart Sapma Arası Bölge', fontweight='bold')
axis = plt.gca()

axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_xticks(range(-4, 5))
axis.text(2, 0.3, f'{result:.3f}', fontsize=14, fontweight='bold')

plt.plot(x, y)

x = np.linspace(-2, 2, 1000)
y = norm.pdf(x)
plt.fill_between(x, y)
axis.arrow(2.5, 0.25, -2, -0.1, width=0.0255)

plt.show()

------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------
Diğer çok karşılaşılan sürekli dağılım "sürekli düzgün dağılım (continous uniform distribution)" 
denilen dağılımdır. Burada dağılımı temsil eden a ve b değerleri vardır. Sürekli 
düzgün dağılımın olasılık yoğunluk fonksiyonu dikdörtgensel bir alandır. Dolayısıyla 
kümülatif dağılım fonksiyonu x değeriyle orantılı bir değer vermektedir. Sürekli 
düzgün dağılımın olasılık yoğunluk fonksiyonu şöyle ifade edilebilir:

f(x) = {
            1 / (b - a)     a < x < b
            0               diğer durumlarda    
       }

------------------------------------------------------------------------------------
Sürekli düzgün dağılım için Python'ın standart kütüphanesinde bir sınıf bulunmamaktadır. 
NumPy'da da böyle bir sınıf yoktur. Ancak SciPy içerisinde stats modülünde uniform 
isimli bir singleton nesne bulunmaktadır. Bu nesneye ilişkin sınıfın yine cdf, ppf, 
pdf ve rvs metotları vardır. Bu metotlar sırasıyl a değerini ve a'dan uzunluğu 
parametre olarak almaktadır. Örneğin:

result = uniform.pdf(15, 10, 10)

Burada aslında a = 10, b = 20 olan bir sürekli düzgün dağılımdaki olasılık yoğunluk 
fonksiyon değeri elde edilmektedir. Tabii aslında 10 ile 20 arasındaki tüm olasılık 
yoğunluk fonksiyon değerleri 1 / 10 olacaktır.


result = uniform.cdf(15, 10, 10)

Burada a = 10, b = 20 olan bir sürekli düzgün dağılımda 15'in solundaki alan elde 
edilecektir. 15 burada orta nokta olduğunda göre elde edilecek bu değer 0.5'tir.
uniform nesnesinin metotlarındaki ikinci parametrenin (loc) a değeri olduğuna ancak 
üçüncü parametrenin a'dan uzaklık belirttiğine (scale) dikkat ediniz.

------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

A = 10
B = 20

x = np.linspace(A - 5, B + 5, 1000)
y = uniform.pdf(x, A, B - A)

plt.title('Continupos Uniform Distribution', fontweight='bold')
plt.plot(x, y)

x = np.linspace(10, 12.5, 1000)  # 12.5'in cdf'sini boyamak icin
y = uniform.pdf(x, A, B - A)
plt.fill_between(x, y)
plt.show()

result = uniform.cdf(12.5, A, B - A)
print(result)                               # 0.25

result = uniform.ppf(0.5, A, B - A)
print(result)                               # 15
------------------------------------------------------------------------------------

Düzgün dağılmış rastgele sayı aslında bizim aşina olduğumuz klasik rastgele üretimidir. 
Örneğin Python standart kütüphanesindeki random modülünde bulnan random fonksiyonu 
0 ile 1 arasında rastgele bir sayı veriyordu. Aslında bu fonksiyon a = 0, b = 1 olan
düzgün dağılımda rastegele sayı veren fonksiyonla tamamne aynıdır. Bnezer biçimde 
NumPy'daki random modülündeki random fonksiyonu 0 ile 1 arasında düzgün dağılmış 
rastgele sayı üretmektedir. Örneğin:

result = uniform.rvs(10, 10, 10)
print(result)  

Burada a = 10, b = 20 olan sürekli düzgün dağılımda 10 tane rastgele noktalı sayı 
elde edilecektir. 

Örneğin 100 ile 200 arasında rastgele 10 tane gerçek sayı üretmek istesek bu işlemi 
şöyle yapmalıyız:

uniform.rvs(100, 200, 10)
------------------------------------------------------------------------------------
"""

# t Dağılımı

"""   
------------------------------------------------------------------------------------
Özellikle güven aralıklarında (confidence interval) ve hipotez testlerinde kullanılan 
diğer önemli bir sürekli dağılım da "t dağılımı (t distribution)" denilen dağılımdır.

t dağılımı standart normal dağılıma oldukça benzemektedir. Bu dağılımın ortalaması 0'dır. 
Ancak standart sapması "serbestlik derecesi (degrees of freedom)" denilen bir değere 
göre değişir. t dağılımının standart sapması sigma = karekök(df / (df - 2)) biçimindedir. 
t dağılımın olasılık yoğunluk fonksiyonu biraz karmaşık bir görüntüdedir. Ancak 
fonksiyon standart normal dağılıma göre "daha az yüksek ve biraz daha şişman" gibi 
gözükmektedir. t dağılımının serbestlik derecesi artırıldığında dağılım standart 
normal dağılıma çok benzer hale gelir. Serbestlik derecesi >= 30 durumunda standart 
normal dağılımla oldukça örtüşmektedir. Yani serbestlik derecesi >= 30 durumunda 
artık t dağılımı kullanmakla standart normal dağılım kullanmak arasında önemli bir 
farklılık kalmamaktadır.

t dağılımı denildiğinde her zaman ortalaması 0 olan standart sapması 1 olan (bu 
konuda bazı ayrıntılar vardır) dağılım anlaşılmaktadır. Tabii t dağılımı da eksende 
kaydırılabilir ve standart sapma değiştirilebilir.

t dağılımı teorik bir dağılımdır. Yukarıda da belirttiğimiz gibi özellikle "güven 
aralıklarının oluşturulması" ve "hipotez testlerinde" kullanım alanı bulmaktadır. 

!!!!!
Bu tür durumlarda anakütle standart sapması bilinmediği zaman örnek standart sapması
anakütle standart sapması olarak kullanılmakta ve t dağılımından faydalanılmaktadır.
!!!!!

t dağılımının önemli bir parametresi "serbestlik derecesi (degrees of freedom)" 
denilen parametresidir. Serbestlik derecesi örneklem büyüklüğünden bir eksik değeri 
belirtir. Örneğin örneklem büyüklüğü 10 ise serbestlik derecesi 9'dur. 

------------------------------------------------------------------------------------
 t dağılımına ilişkin Python standart kütüphanesinde bir sınıf yoktur. NumPy 
kütüphanesinde de t dağılımına ilişkin bir öğe bulunmamaktadır. Ancak SciPy 
kütüphanesindeki stats modülünde t isimli singleton nesne t dağılımı ile işlem yapmak 
için kullanılmaktadır. t isimli singleton nesnenin metotları norm nesnesinin 
metotlarıyla aynıdır. 

Bu fonksiyonlar genel olarak önce x değerlerini sonra serbestlik derecesini, 
sonra da ortalama değeri ve standart sapma değerini parametre olarak almaktadır. 
Ancak yukarıda da belirttiğimiz gibi t dağılımı denildiğinde genel olarak ortalaması 
0, standart sapması 1 olan t dağılımı anlaşılır.

------------------------------------------------------------------------------------
Aşağıdaki programda standart normal dağılım ile 5 serbestlik derecesi ve 30 30 
serbestlik derecesine ilişkin t dağılımlarının olasılık yoğunluk fonksiyonları 
çizdirilmiştir. Burada özellikle 30 serbestlik derecesine ilişkin t dağılımının 
grafiğinin standart normal dağılım grafiği ile örtüşmeye başladığına dikkat ediniz. 


import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_xticks(range(-4, 5))
plt.plot(x, y)

y = t.pdf(x, 1)                 # 1 --> serbestlik derecesi
plt.plot(x, y)

plt.legend(['Standart Normal Dağılım', 't Dağılımı (DOF = 5)', 't dağılımı (DOF = 30)'])

y = t.pdf(x, 30)                # 30 --> serbestlik derecesi
plt.plot(x, y, color='red')     
                    
                # >= 30'dan sonra normal dağılımla hemen hemen aynı grafik oluyor  
plt.show()
------------------------------------------------------------------------------------

Değişik Serbestlik Derecelerine İlişkin t Dağılımı Grafikleri


from scipy.stats import norm,  t
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(15, 10))
plt.title('Değişik Serbestlik Derecelerine İlişkin t Dağılımı Grafikleri', fontweight='bold')
axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

y_norm = norm.pdf(x)
plt.plot(x, y_norm, color='blue')

df_info = [(2, 'red'), (5, 'green'), (10, 'black')]

for df, color in df_info:
    y_t = t.pdf(x, df)
    plt.plot(x, y_t, color=color)
    
plt.legend(['Standart Normal Dağılım'] + [f'{t[0]} Serbestlik Derecesi' for t in df_info], fontsize=14)

plt.show()

------------------------------------------------------------------------------------
Tabii standart normal dağılımla t dağılının olasılık yoğunluk fonksiyonları farklı 
olduğuna göre aynı değerlere ilişkin kümülatif olasılık değerleri de farklı olacaktır. 


from scipy.stats import norm,  t

x = [-0.5, 0, 1, 1.25]
result = norm.cdf(x)
print(result)               # [0.30853754 0.5        0.84134475 0.89435023]

x = [-0.5, 0, 1, 1.25]
result = t.cdf(x, 5)    
print(result)               # [0.31914944 0.5        0.81839127 0.86669189]

------------------------------------------------------------------------------------
"""

# Kesikli (discrete) dağılım
"""  
------------------------------------------------------------------------------------
 Kesikli dağılımlarda X değerleri her gerçek değeri almamaktadır. Dolayısıyla bunların 
fonksiyonları çizildiğinde sürekli fonksiyonlar elde edilemeyecek yalnızca noktalar 
elde edilecektir. Kesikli dağılımlarda X değerlerini onların olasılıklarına eşleyen 
fonksiyonlara "olasılık kütle fonksiyonu (probability mass function)" denilmektedir. 
Sürekli rassal değişkenlerin olasılık yoğunluk fonksiyonları integral hesap için 
kullaılırken kesikli rassal değişkenler için olasılık kütle fonksiyonları doğrudan 
rassal değişkeninin ilgili noktadaki olasılığını vermektedir. Tabii olasılık kütle 
fonksiyonu f(x) olmak üzere her x için f(x) değerlerinin toplamının yine 1 olması 
gerekmektedir.

------------------------------------------------------------------------------------

# Poisson dağılımı

En çok karşılaşılan kesikli dağılımlardan biri "poisson (genellikle "puason" biçiminde 
okumuyor)" dağılımıdır. Bu kesikli dağılım adeta normal dağılımın kesikli versiyonu 
gibidir. Poisson dağılımının olasılık kütle fonksiyonu şöyledir:

    P(X = x) = (e^-lambda * lambda^x) / x!

Buradaki x olasılığını hesaplamak istediğimiz kesikli değeri belirtmektedir. Lamda 
ise ortalama olay sayısını belirtmektedir. Lambda değeri ortalama belirttiği için 
gerçek bir değer olabilir. Ancak x değerleri 0, 1, 2, ... n biçiminde o ve pozitif 
tamsayılardan oluşmaktadır.

Yukarıda da belirttiğimiz gibi poisson dağılımı adeta normal dağılımın kesikli hali 
gibidir. Dolayısıyla doğada da çok karşılaşılan kesikli dağılımlardandır.

Poisson dağılımı için de Python standart kütüphanesinde ya da Numpy ve Pandas 
kütüphanelerinde özel fonksiyonlar ve sınıflar bulunmamaktadır. Ancak SciPy 
kütüphanesinin stats modülü içerisinde poisson isimli bir single nesne ile bu 
dağılımla ilgili işlemler kolaylıkla yapılabilmektedir. 

SciPy'da kesikli dağılımlar üzerinde işlemler yapan singleton nesneler sürekli 
dağılımlarla işlemler yapan single nesnelere kullanım bakımından oldukça benzemektedir. 
Ancak kesikli dağılımlar için fonksiyonun ismi "pdf değil pmf" biçimindedir. Burada 
"pmf" ismi "probability mass function" sözcükleridnen kısaltılmıştır. 

SciPy'daki poisson nesnesinin fonksiyonları genel olarak bizden x değerini ve lambda 
değerini parametre olarak istemektedir. Örneğin maçlardaki gol sayısının poisson 
dağılıma uyduğunu varsayalım. Maçlardaki ortalama gol sayısının 2 olduğunu kabul 
edelim. Bu durumda bir maçta 5 gol olma olasılığı aşağıdaki gibi elde edilebilir:

from scipy.stats import poisson    
result = poisson.pmf(5, 2)
print(result)           # 0.03608940886309672


Yukarıdaki gibi poisson dağılımı sorularında genellikle soruyu soran kişi belli bir 
olayın ortalama gerçekleşme sayısını verir. Sonra kişiden bazı değerleri bulmasını 
ister. Peki soruda "bir maçta 2'den fazla gol olma olasılığı sorulsaydı biz soruyu 
nasıl çözerdik? poisson nesnesi ile cdf fonksiyonunu çağırdığımızda bu cdf fonksiyonu 
bize x değerine kadarki (x değeri de dahil olmak üzere) kümülatif olasılığı verecektir. 
Bu değeri de 1'den çıkartırsak istenen olasılığı elde edebiliriz:

result = 1 - poisson.cdf(2, 2)
print(result)   # 0.3233235838169366

------------------------------------------------------------------------------------
Poisson dağılımında lamda değeri yüksek tutulursa saçılma grafiğinin Gauss 
eğrisine benzediğine dikkat ediniz. 

from scipy.stats import poisson
import matplotlib.pyplot as plt

plt.title('Poisson Distribution with Lambda 100', fontweight='bold')
x = range(0, 200)
y = poisson.pmf(x, 100)

plt.scatter(x, y)
plt.show()

result = poisson.pmf(3, 4)
print(result)

------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------

# Bernoulli dağılımı

Bernoulli dağılımında X değeri 0 ya da 1 olabilir. X = 0 durumu bir olayın olumsuz 
olma ya da gerçekleşmeme olasılığını, X = 1 durumu ise bir olayın olumlu olma ya 
da gerçekleşme olasılığını belirtmektedir. Bu rassal değişkenin yalnızca iki değer 
aldığına dikkat ediniz. Sonucu iki değerden biri olan rassal deneylere 
"Bernoulli deneyleri" de denilmektedir. Bernoulli dağılımının olasılık kütle 
fonksiyonu şöyle ifade edilebilir:

P{X = x} =  {
                p           X = 1 ise
                1 - p       X = 0 ise

            }

Ya da bu olasılık kütle fonksiyonunu aşağıdaki gibi de ifade edebiliriz:

P{X = x} = p^x * (1 - p)^(1 - x)

Burada X = 0 için 1 - p değerinin X = 1 için p değerinin elde edildiğine dikkat ediniz. 


Bernoulli dağılımı için de SciPy kütüphanesinde stats modülü içerisinde bernoulli 
isimli bir singleton nesne bulundurulmuştur. Tabii bu dağılım çok basit olduğu 
için bu nesnenin kullanılması da genellikle gereksiz olmaktadır. bernoulli nesnesinin
ilikin olduğu sınıfın metotları bizden X değerini (0 ya da 1 olabilir) ve X = 1 
için p değerini almaktadır. Örneğin:

from scipy.stats import bernoulli
bernoulli.pmf(0, 0.7)               # 0.3


buradan 0.3 değeri elde edilecektir. 
------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------

# binom dağılım

Binom dağılımında bir Bernoulli deneyi (yani iki sonucu olan bir deney) toplam 
n defa yinelenmektedir. Bu n defa yinelenmede olayın tam olarak kaç defa olumlu
sonuçlanacağının olasılığı hesaplanmak istenmektedir. Dolayısıyla binom dağılımının 
olasılık kütle fonksiyonunda X değerleri 0, 1, 2, ... gibi tamsayı değerler alır. 

Örneğin bir para 5 kez atılıyor olsun. Biz de bu 5 kez para atımında tam olarak 
3 kez Tura gelme olasılığını hesaplamak isteyelim. Burada paranın atılması bir 
Bernoulli deneyidir. Bu deney 5 kez yinelenmiş olumlu kabul ettiğimiz Tura gelme 
durumunun toplamda 3 kez olmasının olasılığı elde edilmek istenmiştir. 
 
Binom dağılımının olasılık kütle fonksiyonu şöyledir:

    P{X = x} = C(n, x) *  p^x  *  (1 - p)^(n - x)


Binom dağılımı için SciPy kütüphanesindeki stats modülünde bulunan binom isimli 
singleton nesne kullanılabilir. Nesnenin kullanılması diğer singleton nesnelere 
benzemektedir. Örneğin ilgili sınıfın pmf fonksiyonu olasılık kütle fonlsiyonunu 
belirtmektedir. 


Örneğin yukarıda belirttiğimiz paranın 5 kez atılmasında tam olarak 3 kez tura 
gelme olsaılığı şöyle elde edilebilir:

from scipy.stats import binom
binom.pmf(3, 5, 0.5)    

Bu fonksiyonlarda birinci parametre "olumlu gerçekleşme sayısını", ikinci parametre 
"toplam deney sayısını"  ve üçüncü parametre de "olumlu gerçekleşme olasılığını" 
belirtmektedir. 
------------------------------------------------------------------------------------
"""

#   Merkezi limit teoremi (central limit theorem)

"""
------------------------------------------------------------------------------------
Bu teoreme göre bir "anakütleden (population)" çekilen belli büyüklükteki örneklerin 
ortalamaları normal dağılmaktadır. Örneğin elimizde 100,000 elemanlı bir anakütle 
olsun. Bu anakütleden 50'lik tüm alt kümeleri yani örnekleri elde edip bunların 
ortalamalarını hesaplayalım. İşte bu ortalamalar normal dağılmaktadır. Bir anakütleden 
alınan altkümelere "örnek (sample)" denilmektedir. Bu işleme de genel olarak 
"örnekleme (sampling)" denir. Örneğimizdeki 100,000 elemanın 50'li alt kümelerinin 
sayısı çok fazladır. O halde deneme için 100,000 elemanlı anakütlenin tüm alt 
kümelerini değil belli sayıda alt kümelerini elde ederek histogram çizebiliriz. Bu 
histogramın teoreme göre Gauss eğrisine benzemesi gerekir. 

------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = np.random.randint(0, POPULATION_RANGE, POPULATION_SIZE)
samples = np.random.choice(population, (NSAMPLES, SAMPLE_SIZE))
samples_means = np.mean(samples, axis=1)

plt.hist(samples_means, bins=50)

plt.show()

------------------------------------------------------------------------------------
Tabii yukarıdaki örneği hiç NumPy kullanmadan tamamen Python standart kütüphanesi 
ile de yapabilirdik.

import random
import statistics

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = random.sample(range(POPULATION_RANGE), POPULATION_SIZE)

samples_means = [statistics.mean(random.sample(population, SAMPLE_SIZE)) for _ in range(NSAMPLES)]

samples_means_mean = statistics.mean(samples_means)

import matplotlib.pyplot as plt

plt.title('Central Limit Theorem', fontweight='bold')
plt.hist(samples_means, bins=50)
plt.show()

------------------------------------------------------------------------------------
Merkezi limit teoremine göre örnek ortalamalarına ilişkin normal dağılımın 
ortalaması anakütle ortalaması ile aynıdır. (Yani bir anakütleden çekilen örnek 
ortalamalarının ortalaması anakütle ortalaması ile aynıdır.)
Aşağıdaki programda bu durum gösterilmiştir.


import numpy as np
import matplotlib.pyplot as plt

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = np.random.randint(0, POPULATION_RANGE, POPULATION_SIZE)
samples = np.random.choice(population, (NSAMPLES, SAMPLE_SIZE))

population_mean = np.mean(population)
samples_means = np.mean(samples, axis=1)
samples_means_mean = np.mean(samples_means)

plt.hist(samples_means, bins=50)

plt.show()

print(f'Anakütle ortalaması: {population_mean}')
print(f'Örnek ortalamalarının Ortalaması: {samples_means_mean}')

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

Peki bir anakütleden alınan örnek ortalamaları normal dağılmaktadır ve bu normal 
dağılımın ortalaması anakütle ortalamasına eşittir. Peki örnek ortalamalarına 
ilişkin normal dağılımın standart sapması nasıldır? İşte merkezi limit teoremine 
göre örnek ortalamalarının standart sapması ------------->

"anakütle kütle standart sapması / karekök(n)" biçimindedir.   n->(örnek büyüklüğü)

Yani örnek ortalamalarının standart sapması anakütle sapmasından oldukça küçüktür 
ve örnek büyüklüğüne bağlıdır. Buradaki n örnek büyüklüğünü belirtmektedir. Örnek 
ortalamalarının standart sapmasına "standard error" de denilmektedir. Görüldüğü 
gibi eğer örnek ortalamalarına ilişkin normal dağılımın standart sapması düşürülmek 
isteniyorsa (başka bir deyişle standart hata azaltılmak isteniyorsa) örnek büyüklüğü
artırılmalıdır. 

------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = np.random.randint(0, POPULATION_RANGE, POPULATION_SIZE)
samples = np.random.choice(population, (NSAMPLES, SAMPLE_SIZE))

population_mean = np.mean(population)
population_std = np.std(population)

samples_means = np.mean(samples, axis=1)
samples_means_mean = np.mean(samples_means)
sample_means_std = np.std(samples_means)

plt.hist(samples_means, bins=50)

plt.show()

print(f'Anakütle ortalaması: {population_mean}')
print(f'Örnek ortalamalarının Ortalaması: {samples_means_mean}')
print(f'Fark: {np.abs(population_mean - samples_means_mean)}')
print("------------------")

print(f'Merkezi limit teroreminden elde edilen örnek ortalamalarının standart sapması: {population_std / np.sqrt(SAMPLE_SIZE)}')
print()
print(f'Örnek ortalamalarının standart sapması: {sample_means_std}')

------------------------------------------------------------------------------------

Merkezi limit teoremine göre eğer anakütleden çekilen örnekler büyükse yani tipik 
olarak n örnek büyüklüğü N ise anakütle büyüklüğü olmak üzere n / N >= 0.05 ise 
bu durumda örneklem dağılımın standart sapması için "anakütle standart sapması / kök(n)" 
değeri "düzeltme faktörü (correction factor)" denilen bir çarpanla çarpılmalıdır. 
Düzeltme faktörü karekök((N - n)/(N -1)) biçimindedir. Bu konunun ayrıntıları için 
başka kaynaklara başvurabilirsiniz. Örneğin ana kütle 100 elemandan oluşuyor olsun. 
Biz 30 elemanlı örnekler çekersek bu düzeltme faktörünü kullanmalıyız. Tabii 
paratikte genellikle n / N değeri 0.05'ten oldukça küçük olmaktadır.

------------------------------------------------------------------------------------

Merkezi limit teoreminde anakütlenin normal dağılmış olması gerekmez. Nitekim 
yukarıdaki örneklerimizde bir anakütleyi "düzgün dağılıma (uniform distribution)" 
ilişkin olacak biçimde oluşturduk. Ancak eğer anakütle normal dağılmamışsa örneklem 
ortalamalarının dağılımının normal olması için örnek büyüklüklerinin belli bir 
değerden büyük olması gerekmektedir. Bu değer tipik olarak >= 30 biçimindedir.
Özet olarak:
    
- Eğer anakütle normal dağılmışsa örnek büyüklüğü ne olursa olsun örneklem 
ortalamalarının dağılımı normaldir. 

- Eğer anakütle normal dağılmamışsa örneklem ortalamalarının dağılımının normal 
olması için örnek büyüklüğünün >= 30 olması gerekir. Tabii n < 30 durumunda yine 
örneklem dağılımı normale benzemektedir ancak kusurlar oluşmaktadır.

Özetle anakütle normal dağılmamışsa örnek büyüklüklerinin artırılması gerekmektedir. 

------------------------------------------------------------------------------------
"""

# Normalliğin Test Edilmesi

"""
------------------------------------------------------------------------------------
 
Sonuç çıkartıcı istatistikte (inferential statisics) bazen anakütlenin normal dağılıp 
dağılmadığını örneğe dayalı olarak test etmek gerebilir. Bunun için anakütleden 
bir örnek alınır. Sonra bu örneğe bakılarak anakütlenin normal dağılıp dağılmadığı 
belli bir "güven düzeyinde (confidence level)" belirlenir. 

Buna "normallik testleri" denilmektedir. Aslında normallik testi gözle de üstünkörü 
yapılabilmektedir. Anakütle içerisinden bir örnek çekip onun histogramını çizersek 
eğer bu histogram Gauss eğrisine benziyorsa biz anakütlenin de normal dağılmış 
olduğu sonucunu gözle tespit edebiliriz. Ancak anakütlenin normal dağılıp 
dağılmadığının tespit edilmesi için aslında "hipotez testleri (hypothesis testing)" 
denilen özel testler kullanılmaktadır. Normal dağılıma ilişkin iki önemli hipotez 
testi vardır: 

"Kolmogorov-Smirnov" testi ve "Shapiro-Wilk" testidir. Bu testlerin istatistiksel 
açıklaması biraz karmaşıktır ve bizim konumuz içerisinde değildir. Ancak bu testler 
SciPy kütüphanesindeki stats modülü içerisinde bulunan fonksiyonlarla yapılabilmektedir. 

Hipotez testlerinde bir hipotez öne sürülür ve bu hipotezin belirli güven düzeyi 
içerisinde doğrulanıp doğrulanmadığına bakılır. Genel olarak bizim doğrulanmasını 
istediğimiz hipoteze H0 hipotezi, bunun tersini belirten yani arzu edilmeyen durumu 
belirten hipoteze de H1 hipotezi denilmektedir. Örneğin normal testindeki H0 ve H1 
hipotezleri şöyle oluşturulabilir:

    H0: Seçilen örnek normal bir anakütleden gelmektedir. 
    H1: Seçilen örnek normal dağılmış bir anakütleden gelmemektedir.
    
------------------------------------------------------------------------------------
scipy.stats modülündeki kstest fonksiyonunun ilk iki parametresi zorunlu parametrelerdir. 
Birinci parametre anakütleden rastgele seçilen örneği alır. İkinci parametre testi 
yapılacak dağılımın kümülatif dağılım fonksiyonunu parametre olarak almaktadır. 
Ancak bu parametre kolaylık olsun diye yazısal biçimde girilebilmektedir. Normallik 
testi için bu parametre 'norm' ya da norm.cdf biçiminde girilebilir. 

------------------------------------------------------------------------------------
kstest fonksiyonu çağrıldığktan sonra bize "isimli bir demet (named tuple) verir. 
Demetin birinci elemanı test istatistiğini, ikinci elemanı p değerini belirtir. 
Bizim burada yapmamız gereken bu p değerinin kendi seçtiğimiz belli bir kritik
değerden büyük olup olmadığına bakmaktır. Bu kritik değer tipik olarak 0.05 olarak 
alınmaktadır. Ancak testi daha katı yapacaksanız bu değeri 0.01 gibi küçük tutabilirsiniz.

Eğer "p değeri belirlediğimiz kritik değereden (0.05) büyükse H0 hipotezi kabul edilir, 
H1 hipotezi reddedilir. Eğer p değeri bu kritik değerden küçükse H0 hipotezi reddelip, 
H1 hipotezi kabul edilmektedir. Yani özetle bu p değeri 0.05 gibi bir kritik değerden 
büyükse örnek normal dağılmış bir anakütleden gelmektedir, 0.05 gibi bir kritik 
değerden küçükse örnek normal dağılmamış bir anakütleden gelmektedir.

Aşağıdaki örnekte normal dağılmış bir anakütleden ve düzgün dağılmış bir anakütleden 
rastgele örnekler çekilip kstest fonksiyonuna sokulmuştur.
    
    
from scipy.stats import norm, uniform, kstest

sample_norm = norm.rvs(size=1000)

result = kstest(sample_norm, 'norm')
print(result.pvalue)        # 1'e yakın bir değer             

sample_uniform = uniform.rvs(size=1000) 

result = kstest(sample_uniform, norm.cdf)
print(result.pvalue)        # 0'a çok yakın bir değer

------------------------------------------------------------------------------------    
kstest fonksiyonunda ikinci parametreye 'norm' girildiğinde birinci parametredeki 
değerler ortalaması 0, standart sapması 1 olan standart normal dağılıma uygun değerler 
olmalıdır. Eğer ortalama ve standart sapması farklı bir normal dağılım testi 
yapılacaksa dağılımın parametreleri de ayrıca args parametresiyle verilmelidir. 
Örneğin biz ortalaması 100 standart sapması 15 olan bir dağılıma ilişkin test yapmak 
isteyelim. Bu durumda test aşağıdaki gibi yapılmalıdır. 


from scipy.stats import norm, uniform, kstest

sample_norm = norm.rvs(100, 15, size=1000)  # ort=100 std'si 15 olan rastgele değerler

result = kstest(sample_norm, 'norm', args=(100, 15))
print(result.pvalue)                   

sample_uniform = uniform.rvs(100, 100, size=1000) # 100 ile 200 arasında rastgele değerler

result = kstest(sample_uniform, norm.cdf, args=(100, 100))
print(result.pvalue)

------------------------------------------------------------------------------------  
Shapiro-Wilk testi de tamamen benzer biçimde uygulanmaktadır. Ancak bu fonksiyonun 
kullanılması daha kolaydır. Bu fonksiyonun tek bir parametresi vardır. Bu parametre 
anakütleden çekilen örneği belirtir. Buradaki normal dağılım herhangi bir ortalama ve 
standart sapmaya ilişkin olabilir. Yani bizim dağılım değerlerini ortalaması 0, 
standart sapması 1 olacak biçimde ölçeklendirmemiz gerekmemektedir. 

Ayrıca anakütleden çekilen örnekler küçükse (tipik olarak <= 50) Shapiro-Wilk testi 
Kolmogorov-Simirnov testine göre daha iyi bir sonucun elde edilmesine yol açmaktadır. 
Yani örneğiniz küçükse Shapiro-Wilk testini tercih edebilirsiniz. 

Aşağıdaki örnekte ortalaması 100, standart sapması 15 olan normal dağılmış ve düzgün 
dağılmış bir anakütleden 100'lük bir örnek seçilip Shapiro-Wilk testine sokulmuştur. 
Buradan elde edine pvalue değerlerine dikkat ediniz. 


from scipy.stats import norm, uniform, shapiro

sample_norm = norm.rvs(100, 15, size=100)

result = shapiro(sample_norm)
print(result.pvalue)                   

sample_uniform = uniform.rvs(100, 100, size=100) 

result = shapiro(sample_uniform)
print(result.pvalue)

------------------------------------------------------------------------------------  
"""

# Örnekten Hareketle Anakütle Parametrelerinin Tahmin Edilmesi

"""
------------------------------------------------------------------------------------  
İstatistikte örneğe dayalı olarak anakütlenin ortalamasını (ve/veya standart sapmasını) 
tahmin etmeye "parametre tahmini (parameter estimation)" denilmektedir. Parametre 
tahmini "noktasal olarak (point estimate)" ya da "aralıksal olarak (interval estimate)" 
yapılabilmektedir. Anakütle ortalamasının aralıksal tahminine "güven aralıkları 
(confidence interval)" da denilmektedir. Güven aralıkları tamamen merkezi limit 
teroremi kullanılarak oluşturulmaktadır.

------------------------------------------------------------------------------------  
Bizim anakütle ortalamasını bilmediğimizi ancak anakütle standart sapmasını bildiğimizi 
varsayalım. (Genellikle aslında anakütle standart sapmasını da bilmeyiz. Ancak 
burada bildiğimizi varsayıyoruz.) Bu anakütleden rastgele bir örnek seçtiğimizde 
o örneğin ortalamasına bakarak anakütle ortalamasını belli bir aralıkta belli bir 
güven düzeyinde tahmin edebiliriz. 

Şöyle ki: Örneğin seçtiğimiz güven düzeyi %95 olsun. Bu durumda bizim örneğimiz 
örnek ortalamalarının dağılımında en kötü olasılıkla soldan 0.025 ve sağdan 0.975 
kümülatif olasılığa karşı gelen x değerlerinden biri olabilir. Tabii seçtiğimiz 
örneğin ortalamasının bu aralıkta olma olasılığı %95'tir Bu durumda yapacağımız 
şey örnek ortalamasının örneklem dağılımına göre seçtiğimiz örneğin ortalamasının 
%47.5 soluna ve %47.5 sağına ilişkin değerlerin elde edilmesidir. Bu durumda anakütle 
ortalaması %95 güven düzeyi içerisinde bu aralıkta olacaktır. Tabii aslında bu 
işlemi daha basit olarak "rastgele elde ettiğimiz örneğin ortalamasını normal 
dağılımın merkezine alarak soldan 0.025 ve sağdan 0.975 kümülatif olasılık 
değerlerine karşı gelen noktaların elde edilmesi yoluyla da yapabiliriz.  

------------------------------------------------------------------------------------  
Örneğin standart sapması 15 olan bir anakütleden rastgele 60 elemanlık bir örnek 
elde etmiş olalım. Bu örneğin ortalamasının 109 olduğunu varsayalım. Bu durumda 
%95 güven düzeyi içerisinde anakütle ortalamasına ilişkin güven aralıkları 
aşağıdaki gibi elde edilebilir:

    
import numpy as np
from scipy.stats import norm

sample_size = 60
population_std = 15
sample_mean = 109
sampling_mean_std = population_std / np.sqrt(sample_size)

lower_bound = norm.ppf(0.025, sample_mean, sampling_mean_std)
upper_bound = norm.ppf(0.975,  sample_mean, sampling_mean_std)

print(f'{lower_bound}, {upper_bound}')              # 105.20454606435501, 112.79545393564499


Burada biz anakütlenin standart sapmasını bildiğimiz için örnek ortalamalarına 
ilişkin normal dağılımın standart sapmasını hesaplayabildik. Buradan elde ettiğimiz 
güven aralığı şöyle olmaktadır:

105.20454606435501, 112.79545393564499
------------------------------------------------------------------------------------ 

Güven düzeyini yükseltirsek güven aralığının genişleyeceği açıktır. Örneğin bu 
problem için güven düzeyini %99 olarak belirlemiş olalım:

import numpy as np
from scipy.stats import norm

sample_size = 60
population_std = 15
sample_mean = 109
sampling_mean_std = population_std / np.sqrt(sample_size)

lower_bound = norm.ppf(0.005, sample_mean, sampling_mean_std)
upper_bound = norm.ppf(0.995,  sample_mean, sampling_mean_std)

print(f'{lower_bound}, {upper_bound}')     # 104.01192800234102, 113.98807199765896


Burada güven aralığının aşağıdaki gibi olduğunu göreceksiniz:

104.01192800234102, 113.98807199765896

Gördüğünüz gibi aralık büyümüştür.

------------------------------------------------------------------------------------  
Anımsanacağı gibi örnek ortalamalarına ilişkin dağılımın standart sapmasın "standart hata 
(standard error)" deniliyordu. Örnek ortalamalarına ilişkin dağılımın standart sapması 
azaltılırsa (yani standart hata düşürülürse) değerler ortalamaya yaklaşacağına göre 
güven aralıkları da daralacaktır. O halde anakütle ortalamasını tahmin ederken 
büyük örnek seçmemiz güven aralıklarını daraltacaktır. Aşağıdaki örnekte yukarıdaki 
problemin 30'dan 100'e kadar beşer artırımla örnek büyüklükleri için %99 güven 
düzeyinde güven aralıkları elde edilmiştir. Elde edilen aralıklar şöyledir:

import numpy as np
from scipy.stats import norm

population_std = 15
sample_mean = 109

for sample_size in range(30, 105, 5):
    
    sampling_mean_std = population_std / np.sqrt(sample_size)
    
    lower_bound = norm.ppf(0.025, sample_mean, sampling_mean_std)
    upper_bound = norm.ppf(0.975,  sample_mean, sampling_mean_std)
    
    print(f'sample size: {sample_size}: [{lower_bound}, {upper_bound}]')  

    
sample size: 30: [103.63241756884852, 114.36758243115148]
sample size: 35: [104.03058429805395, 113.96941570194605]
sample size: 40: [104.35153725771579, 113.64846274228421]
sample size: 45: [104.61738729711709, 113.38261270288291]
sample size: 50: [104.84228852695097, 113.15771147304903]
sample size: 55: [105.03577765357056, 112.96422234642944]
sample size: 60: [105.20454606435501, 112.79545393564499]
sample size: 65: [105.3534458105975, 112.6465541894025]
sample size: 70: [105.48609245861904, 112.51390754138096]
sample size: 75: [105.60524279777148, 112.39475720222852]
sample size: 80: [105.71304047283782, 112.28695952716218]
sample size: 85: [105.81118086644236, 112.18881913355763]
sample size: 90: [105.9010248384772, 112.0989751615228]
sample size: 95: [105.98367907149301, 112.01632092850699]
sample size: 100: [106.06005402318992, 111.93994597681008]


Buradan da gördüğünüz gibi örneği büyüttüğümüzde güven aralıkları daralmakta ve 
anakütle ortalaması daha iyi tahmin edilmektedir. Örnek büyüklüğünün artırılması 
belli bir noktaya kadar aralığı iyi bir biçimde daraltıyorsa da belli bir noktadan 
sonra bu daraltma azalmaya başlamaktadır. Örneklerin elde edilmesinin belli bir 
çaba gerektirdiği durumda örnek büyüklüğünün makul seçilmesi önemli olmaktadır.

------------------------------------------------------------------------------------  
Anımsanacağı gibi anakütle normal dağılmamışsa merkezi limit teoreminin yeterli 
bir biçimde uygulanabilmesi için örneklerin büyük olması (tipik olarak >= 30) 
gerekiyordu. O halde güven aralıklarını oluştururken eğer anakütle normal dağılmamışsa
bizim örnekleri >= 30 biçiminde seçmemiz uygun olur.

------------------------------------------------------------------------------------  
Örnekten hareketle anakütle ortalamaları aslında tek hamlede norm nesnesinin ilişkin 
olduğu sınıfın interval metoduyla da elde edilebilmektedir. interval metodunun 
parametrik yapısı şöyledir:

interval(confidence, loc=0, scale=1)

Metodun birinci parametresi olan confidence güven düzeyini belirtmektedir. Örneğin 
%95 güven düzeyi için bu parametre 0.95 girilmelidir. Metodun ikinci ve üçüncü 
parametreleri örnek ortalamalarının dağılımına ilişkin ortalama ve standart sapmayı 
belirtir. Tabii ikinci parametre elde etmiş olduğumuz örneğin ortalaması olarak 
girilmelidir. Metot güven aralığını belirten bir demetle geri döner. Demetin ilk 
elemanı lower_bound ikinci elemanı upper_bound değerlerini vermektedir.

Bu durumda yukarıdaki problemi interval metoduyla aşağıdaki gibi de çözebiliriz:

import numpy as np
from scipy.stats import norm

sample_size = 60
population_std = 15
sample_mean = 109

sampling_mean_std = population_std / np.sqrt(sample_size)
lower_bound, upper_bound = norm.interval(0.95, sample_mean, sampling_mean_std)
print(f'{lower_bound}, {upper_bound}')     

------------------------------------------------------------------------------------  
Aşağıdaki örnekte ortalaması 100, standart sapması 15 olan normal dağılıma uygun 
rastgele 1,000,000 değer üretilmiştir. Bu değerlerin anakütleyi oluşturduğu varsayılmıştır. 
Sonra bu anakütle içerisinden rastgele 60 elemanlık bir örnek elde edilmiştir. Bu 
örneğe dayanılarak ana kütle ortalaması norm nesnesinin interval metoduyla elde 
edilip ekrana yazdırılmıştır.


import numpy as np
from scipy.stats import norm

POPULATION_SIZE = 1_000_000
SAMPLE_SIZE = 60

population = norm.rvs(100, 15, POPULATION_SIZE)

population_mean = np.mean(population)
population_std = np.std(population)

print(f'population mean: {population_mean}')
print(f'population std: {population_std}')
print("----------------------")

sample = np.random.choice(population, SAMPLE_SIZE)
sample_mean = np.mean(sample)
sampling_mean_std = population_std / np.sqrt(SAMPLE_SIZE)

print(f'sample mean: {sample_mean}')


lower_bound, upper_bound = norm.interval(0.95, sample_mean, sampling_mean_std)

print(f'[lower_bound= {lower_bound}, upper_bound= {upper_bound}]')

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
"""
"""
------------------------------------------------------------------------------------  
Biz yukarıdaki örneklerde güven aralıklarını oluştururken anakütle standart sapmasının 
bilindiğini varsaydık. Halbuki genellikle anakütle ortalamasının bilinmediği durumda 
anakütle standart sapması da bilinmemektedir. Peki bu durumda örnekten hareketle 
anakütle ortalamasının aralık tahmini nasıl yapılacaktır? İşte bu durumda çektiğimiz 
örneğin standart sapması anakütlenin standart sapması gibi işleme sokulmaktadır.

Ancak dağılım olarak normal dağılım değil t dağılımı kullanılmaktadır. Zaten Gosset 
t dağılımını tamamen böyle bir problemi çözerken geliştirmiştir. Yani t dağılımı 
zaten "anakütle standart sapmasının bilinmediği durumda örneğin standart sapmasının 
anakütle standart sapması olarak alınmasıyla" elde edilen bir dağılımdır. t dağılımının 
serbestlik derecesi denilen bir değere sahip olduğunu anımsayınız. Serbestlik 
derecesi örnek büyüklüğünün bir eksik değeridir. Ayrıca 30 serbestlik derecesinden 
sonra zaten t dağılımının normal dağılıma çok benzediğini de belirtmiştik.

Çektiğimiz örneğin standart sapmasını anakütle standart sapması olarak kullanırken 
örneğin standrat sapması N'e değil (N - 1)'e bölünerek hesaplanmalıdır. Burada 
bölmenin neden (N - 1)'e yapıldığının açıklaması biraz karmaşıktır. Burada bu 
konu üzerinde durmayacağız. Ancak Internet'te bu konuyu açıklayan pek çok kaynak 
bulunmaktadır.İstatistikte çekilen örneklerin standart sapmaları genellikle sigma 
sembolü ile değil s harfiyle belirtilmektedir.

Anımsanacağı gibi pek çok kütüphanede standart sapma ya da varyans hesaplanırken 
bölmenin neye yapılacağına "ddof (delta degrees of freedom)" deniyordu. Standart 
sapma ya da varyans hesabı yapan fonksiyonların ddof parametreleri vardı. NumPy'da 
bu ddof parametresi default 0 iken Pandas'da 1'dir. Bu ddof parametresi (N - değer)
'deki değeri belirtmektedir. Yani ddof = 0 ise bölme N'e ddof = 1 ise bölme 
(N - 1)'e yapılmaktadır. 

------------------------------------------------------------------------------------  
sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948, 87.1273706 ,  76.37932669,  87.99167058,  95.16206704,
    101.78211828,  80.71674993, 126.3793041 , 105.07860807, 98.4475209 , 124.47749601,  82.79645255,  82.65166373, 92.17531189, 
    117.31491413, 105.75232982,  94.46720598, 100.3795159 ,  94.34234528,  86.78805744,  97.79039692, 81.77519378, 117.61282039, 
    109.08162784, 119.30896688, 98.3008706 ,  96.21075454, 100.52072909, 127.48794967, 100.96706301, 104.24326515, 101.49111644])

Anakütlein standart sapmasının da bilinmediğini varsayalım. Bu değerlerden hareketle 
%95 güven düzeyinde güven aralığını şöyle oluşturabiliriz:

import numpy as np
from scipy.stats import t

sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948, 87.1273706 ,  76.37932669,  
                87.99167058,  95.16206704, 101.78211828,  80.71674993, 126.3793041 , 105.07860807, 
                98.4475209 , 124.47749601,  82.79645255,  82.65166373, 92.17531189, 117.31491413, 
                105.75232982,  94.46720598, 100.3795159 ,  94.34234528,  86.78805744,  97.79039692, 
                81.77519378, 117.61282039, 109.08162784, 119.30896688, 98.3008706 ,  96.21075454, 
                100.52072909, 127.48794967, 100.96706301, 104.24326515, 101.49111644])

sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
sampling_mean_std = sample_std / np.sqrt(len(sample))

lower_bound = t.ppf(0.025, len(sample) - 1, sample_mean, sampling_mean_std)
upper_bound = t.ppf(0.975, len(sample) - 1, sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')


Burada örneğin standart sapmasını hesaplarken ddof=1 kullandığımıza dikkat ediniz. 
Güven aralıkları normal dağılım kullanılarak değil t dağılımı kullanılarak elde 
edilmiştir. t dağılımındaki serbestlik derecesinin (ppf fonksiyonun ikinci parametresi) 
örnek büyüklüğünün bir eksik değeri olarak alındığını anımsayınız.

Serbestlik derecesi 30'dan sonra artık t dağılımın normal dağılımla örtüşmeye 
başladığını anımsayınız. Buradaki örneğimizde örnek büyüklüğü 35'tir. Örnek 
büyüklüğü >= 30 durumunda t dağılı ile normal dağılım birbirine çok benzediği 
için aslında bu örnekte t dağılımı yerine normal dağılım da kullanabilirdi.

------------------------------------------------------------------------------------  
Tıpkı sscipy.stats modülündeki norm nesnesinde olduğu gibi t nesnesinin de ilişkin 
olduğu snıfın interval isimli bir metodu bulunmaktadır. Bu metot zaten doğrudan 
t dağılımını kullanarak güven aralıklarını hesaplamaktadır. interval metodunun 
parametrik yapısı şöyledir:

interval(confidence, df, loc=0, scale=1)

Buradaki confidence paarametresi yine "güven düzeyini (confidence level)" belirtmektedir. 
df parametresi serbestlik derecesini belirtir. loc ve scale parametreleri de sırasıyla 
ortalama ve standart sapma değerlerini belirtmektedir. Burada loc parametresine 
biz örneğimizin ortalamasını, scale parametresine de örneklem dağılımının standart 
sapmasını girmeliyiz. Tabii örneklem dağılımının standart sapması yine örnekten 
hareketle elde edilecektir. Metot yine güven aralığının alt ve üst değerlerini bir 
demet biçiminde geri döndürmektedir.


import numpy as np
from scipy.stats import t

sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948, 87.1273706 ,  76.37932669,  
                   87.99167058,  95.16206704, 101.78211828,  80.71674993, 126.3793041 , 105.07860807, 
                   98.4475209 , 124.47749601,  82.79645255,  82.65166373, 92.17531189, 117.31491413, 
                   105.75232982,  94.46720598, 100.3795159 ,  94.34234528,  86.78805744,  97.79039692, 
                   81.77519378, 117.61282039, 109.08162784, 119.30896688, 98.3008706 ,  96.21075454, 
                   100.52072909, 127.48794967, 100.96706301, 104.24326515, 101.49111644])

sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
sampling_mean_std = sample_std / np.sqrt(len(sample))

lower_bound = t.ppf(0.025, len(sample) - 1, sample_mean, sampling_mean_std)
upper_bound = t.ppf(0.975, len(sample) - 1, sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')
print("---------------------------------")

lower_bound, upper_bound = t.interval(0.95, len(sample) - 1, sample_mean, sampling_mean_std)
print(f'[{lower_bound}, {upper_bound}]')

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  

Merkezi limit teoremine göre eğer ana kütle normal dağılmamışsa ancak n >= 30 koşulunu 
sağlayan örneklem dağılımlarının normal dağıldığı kabul edilmektedir. Yani 
örneklerimizdeki gibi anakütle ortalamasının tahmin edilmesi ve güven aralıklarının 
oluşturulması için şu iki koşuldan en az biri sağlanmalıdır:

1) Anakütle normal dağılmıştır ve örneklem dağılımı için n < 30 durumu söz konusudur.
2) Anakütle normal dağılmamıştır ve örneklem dağılımı için n >= 30 durumu söz konusudur.

                                
Pekiyi örneğimiz küçükse (tipik oalrak < 30) ve ana kütle normal dağılmamışsa güven 
aralıklarını oluşturamaz mıyız? İşte bu tür durumlarda güven aralıklarının oluşturulması 
ve bazı hipotez testleri için "parametrik olmayan (nonparametric) yöntemler kullanılmaktadır. 
Ancak genel olarak parametrik olmayan yöntemler parametrik yöntemlere göre daha 
daha az güvenilir sonuçlar vermektedir. 
------------------------------------------------------------------------------------  
"""



#  --------------------- Verilerin Kullanıma Hazır Hale Getirilmesi ---------------------

"""
------------------------------------------------------------------------------------  
Veriler toplandıktan sonra hemen işleme sokulamayabilir. Veriler üzerinde çeşitli 
ön işlemler yapmak gerekebilir. Bu ön işlemlere "verilerin hazır hale getirilmesi 
(data preparation)" denilmektedir. Biz bu bölümde verilerin hazır hale getirilmesi 
için bazı temel işlemler üzerinde duracağız. Diğer bazı işlemler başka bölümlerde 
ele alınacaktır. Ancak verilerin kullanıma hazır hale getirilmesi tipik olarak 
aşağıdaki gibi süreçleri içermektedir:

1) Verilerin Temizlenmesi (Data Cleaning): Veriler eksiklikler içerebilir ya da 
geçersiz değerler içerebilir. Bazen de aşırı uç değerlerden (outliers) kurtulmak 
gerekebilir. Bu faaliyetlere verilerin temizlenmesi denilmektedir. Kusurlu veriler 
yapılan analizleri olumsuz yönde etkilemektedir.


2) Özellik seçimi (Feature Selection): Veri kümelerindeki tüm sütunlar bizim için 
anlamlı ve gerekli olmayabilir. Gereksiz sütunların atılıp gereklilerin alınması 
faaliyetine "özellik seçimi" denilmektedir. Örneğin bir veri tablosundaki kişinin 
"adı soyadı" sütunu veri analizi açısından genellikle (ama her zaman değil) bir 
fayda sağlamamaktadır. Bu durumda bu sütunların atılması gerekir. Bazen veriler 
tamamen geçersiz bir durum halinde de karşımıza gelebilmektedir. Örneğin oransal 
bir ölçeğe sahip sütunda yanlışlıkla kategorik bir veri bulunabilir.


3) Verilerin Dönüştürülmesi (Data Transformation): Kategorik veriler, tarih ve 
zaman verileri gibi veriler, resimler gibi veriler doğrudan işleme sokulamazlar. 
Bunların sayısal biçime dönüştürülmesi gerekir. Bazen veri kümesindeki sütunlarda 
önemli skala farklılıkları olabilmektedir. Bu skala farklılıkları algoritmaları 
olumsuz etkileyebilmektedir. İşte sütunların skalalarını birbirine benzer hale 
getirme sürecine "özellik ölçeklemesi (feature scaling)" denilmektedir.


4) Özellik Mühendisliği (Feature Engineering): Özellik mühendisliği veri tablosundaki 
sütunlardan olmayan başka sütunların oluşturulması sürecine denilmektedir. Yani 
özellik mühendisliği var olan bilgilerden hareketle önemli başka bilgilerin elde
edilmesidir. Örneğinin kişinin boy ve kilosu biliniyorsa biz vücut kitle endeksini 
tabloya ekleyebiliriz.


5) Boyutsal Özellik İndirgemesi (Dimentionality Feature Reduction): Veri kümesinde 
çok fazla sütun olmasının pek çeşitli dezavantajı olabilmektedir. Örneğin bu tür 
durumlarda işlem yükü artabilir. Gereksiz sütunlar kestirimi sürecini olumsuz
biçimde etkileyebilir. Fazla sayıda sütun kursumuzun ilerleyen zamanalarında sıkça 
karşılaşacağımız "overfitting" denilen yanlış öğrenmelere yol açabilir. O zaman 
sütunların sayısının azaltılması gerekebilir. İşte n tane sütunun k < n olmak üzere
k tane sütun haline getirilmesi sürecine boyutsal özellik indirgemesi denilmektedir. 
Bu konu kursumuzda ileride ayrı bir bölümde ele alınacaktır.


6) Verilerin Çoğaltılması (Data Augmentation): Elimizdeki veriler (veri kümesindeki 
satırlar) ilgili makine öğrenmesi yöntemini uygulayabilmek için sayı bakımından 
ya da nitelik bakımından yetersiz olabilir.  Eldeki verilerle (satırları kastediyoruz)
yeni verilerin oluşturulması (yeni satırların oluşturulması) sürecine "verilerin 
çoğaltılması (data augmentation)" denilmektedir. Örneğin bir resimden döndürülerek 
pek çok resim elde edilebilir. Benzer biçimde örneğin bir resmin çeşitli kısımlarından
yeni resimler oluşturulabilir. Özellik mühendisliğinin "sütun eklemeye yönelik", 
verilerin çoğaltılmasının ise "satır eklemeye yönelik" bir süreç olduğuna dikkat ediniz.

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
CSV dosyalarında iki virgül arasında hiçbir değer yoksa bu eksik veri anlamına 
geliyor olabilir. Böyle CSV dosyalarını Pandas'ın read_csv fonksiyonuyla okursak 
NaN (Not a Number) denilen özel numpy.float64 değerini elde ederiz. Örneğin "person.csv" 
isimli dosya şu içeriği sahip olsun:

Adı Soyadı,Kilo,Boy,Yaş,Cinsiyet
Sacit Bulut,78,172,34,Erkek
Ayşe Er,67,168,45,Kadın
Ahmet San,,182,32,Erkek
Macit Şen,98,156,65,Erkek
Talat Demir,85,,49,Erkek

Bu  dosyayı şöyle okuyalım:

import pandas as pd

df = pd.read_csv('person.csv')

Şöyle bir çıktı elde ederiz:

        Adı Soyadı  Kilo    Boy  Yaş Cinsiyet
    0  Sacit Bulut  78.0  172.0   34    Erkek
    1      Ayşe Er  67.0  168.0   45    Kadın
    2    Ahmet San   NaN  182.0   32    Erkek
    3    Macit Şen  98.0  156.0   65    Erkek
    4  Talat Demir  85.0    NaN   49    Erkek

Bir CSV dosyasında özel bazı sözcükler de eksik veri anlamına gelebilmektedir. 
Ancak hangi özel sözcüklerin eksik veri anlamına geldiği CSV okuyucuları arasında 
farklılıklar gösterebilmektedir. Örneğin Pandas'ın read_csv fonksiyonu şu özel 
sözcükleri "eksik veri" gibi ele almaktadır: NaN, '', '#N/A', '#N/A' 'N/A', '#NA', 
'-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 
'NULL', 'NaN', 'None', 'n/a', 'nan', ‘null’. CSV dosyalarında en çok karşımıza çıkan 
eksik veri gösterimlri şunlardır: '', NaN, nan, NA, null. 

read_csv fonksiyonu ayrıca na_values isimli parametresi yoluyla programcınn istediği 
yazıları da eksik veri olarak ele alabilmektedir. Bu parametreye yazılardan oluşan 
dolaşılabilir bir nesne girilmeldir. Örneğin:

df = pd.read_csv('person.csv', na_values=['NE'])

Burada read_csv yukarıdakilere ek olarak 'NE' yazısını da eksik olarak ele alacaktır. 

read_csv fonksiyonunun keep_default_na parametresi False olarak girilirse 
(bu parametrenin default değeri True biçimdedir) bu durumda yukarıda belirttiğimiz eksik 
veri kabul edilen yazılar artık eksik veri olarak kabul edilmeyecek onlara normal 
yazı muamalesi yapılacaktır. 

read_csv fonksiyonu bir süredir yazısal olan sütunların dtype özelliğini "object" 
olarak tutmaktadır. Yani bu tür sütunların her elemanı farklı türlerden olabilir. 
Bu tür sütunlarda NaN gibi eksik veriler söz konusu olduğunda read_csv fonksiyonu 
yine onu np.float64 NaN değeri olarak ele almaktadır.

Eksik veri işlemleri NumPy kütüphanesi ile loadtxt fonksiyonu kullanılarak da yapılabilir. 
Ancak loadtxt fonksiyonunun eksik verileri ele almak için kullanılması çok zahmetlidir. 
Bu nedenle biz kursumuzda CSV dosyalarını genellikle Pandas'ın read_csv fonksiyonu 
ile okuyacağız. 

------------------------------------------------------------------------------------  
Eksik verilerle çalışırken ilk yapılması gereken şey "eksikliğin analiz edilmesidir". 
Yani kaç tane eksik veri vardır? Kaç satırda ve hangi sütunlarda eksik veriler bulunmaktadır? 
Eksik veriler toplam verilerin yüzde kaçını oluşturmaktadır? Gibi sorulara yanıtlar 
aranmalıdır.

Eksik verilerin ele alınmasında iki temel strateji vardır:

1) Eksik verilerin bulunduğu satırı (nadiren de sütunu) tamamen atmak
2) Eksik verilerin yerine başka değerler yerleştirmek (imputation). 
                                                                   
------------------------------------------------------------------------------------  
Eksik verilerin bulunduğu satırın atılması yönteminde şunlara dikkat edilmelidir:

a) Eksik verili satırlar atıldığında elde kalan veri kümesi çok küçülecek midir?
b) Eksik verili satırlar atıldığında elde kalan veri kümesi yanlı (biased) hale gelecek midir?

Eğer bu soruların yanıtı "hayır" ise eksik verilerin bulunduğu satırlar tamamen atılabilir. 

Eksik veriler yerine bir verinin doldurulması işlemine İngilizce "imputation" denilmektedir. 
Eğer eksik verilerin bulunduğu satır (nadiren de sütun) atılamıyorsa "imputation" 
uygulanmalıdır.

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
 DataFrame nesnesi df olmak üzere, eksik veri analizinde şu kalıpları kullanabilirsiniz:

1) Sütunlardaki eksik verilerin miktarları şöyle bulunabilir:

df.isna().sum() ya da pd.isna(df).sum()

Pandas'taki isna fonksiyonu aynı zamanda DataFrame ve Series sınıflarında bir metot 
biçiminde de bulunmaktadır. isna bize bool türden bir DataFrame ya da Series nesnesi 
vermektedir. bool üzerinde sum işlemi yapıldığında False değerler 0 olarak, True
değerler 1 olarak işleme girer. Dolayısıyla yularıdaki işlemlerde biz sütunlardaki 
eksik veri sayılarını elde etmiş oluruz. isna fonksiyonunun diğer bir ismi isnull 
biçimindedir.

------------------------------------------------------------------------------------  
2) Eksik verilerin toplam sayısı şöyle bulunabilir:

df.isna().sum().sum() ya da pd.isna(df).sum().sum() 

isna fonksiyonu (ya da metodu) sütunsal temelde eksik verilerin sayılarını verdiğine 
göre onların toplamı da toplam eksik verileri verecektir.

------------------------------------------------------------------------------------  
3) Eksik verilerin bulunduğu satır sayısı şöyle elde edilebilir:

pd.isna(df).any(axis=1).sum() ya da df.isna().any(axis=1).sum()

any fonksiyonu ya da metodu bir eksen parametresi alarak satırsal ya da sütunsal 
işlem yapabilmektedir. any "en az bir True değer varsa True değerini veren hiç 
True değer yoksa False değerini veren" bir fonksiyondur. 

Yukarıdaki ifadede biz önce isna fonksiyonu ile eksik verileri matrissel bir biçimde 
DataFrame olarak elde ettik. Sonra da onun satırlarına any işlemi uyguladık. 
Dolayısıyla "en az bir eksik veri olan satırların" sayısını elde etmiş olduk.

------------------------------------------------------------------------------------  
4) Eksik verilerin bulunduğu satır indeksleri şöyle elde edilebilir:

df.index[pd.isna(df).any(axis=1)] ya da df.loc[df.isna().any(axis=1)].index

------------------------------------------------------------------------------------  
5) Eksik verilerin bulunduğu sütun isimleri şöyle elde edilebilir:

missing_columns = [name for name in df.columns if df[name].isna().any()]

Burada liste içlemi kullandık. Önce sütun isimlerini elde edip o sütun bilgilerini 
doğrudan indesklemeyle elde ettik. Sonra o sütunda en az bir eksik veri varsa o 
sütunun ismini listeye ekledik.

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
import pandas as pd

df = pd.read_csv('melb_data.csv')

missing_columns = [colname for colname in df.columns if df[colname].isna().any()]
print(f'Eksik verilen bulunduğu sütunlar: {missing_columns}', end='\n\n')

missing_column_dist = df.isna().sum()
print('Eksik verilerin sütunlara göre dağılımı:')
print(missing_column_dist, end='\n\n')

missing_total = df.isna().sum().sum()
print(f'Eksik verilen toplam sayısı: {missing_total}')

missing_ratio = missing_total / df.size
print(f'Eksik verilen oranı: {missing_ratio}')

missing_rows = df.isna().any(axis=1).sum()
print(f'Eksik veri bulunan satırların sayısı: {missing_rows}')

missing_rows_ratio = missing_rows / len(df)
print(f'Eksik veri bulunan satırların oranı: {missing_rows_ratio}')


---------- Elde Edilen Çıktı ----------

Eksik verilen bulunduğu sütunlar: ['Car', 'BuildingArea', 'YearBuilt', 'CouncilArea']

Eksik verilerin sütunlara göre dağılımı:
Suburb              0
Address             0
Rooms               0
Type                0
Price               0
Method              0
SellerG             0
Date                0
Distance            0
Postcode            0
Bedroom2            0
Bathroom            0
Car                62
Landsize            0
BuildingArea     6450
YearBuilt        5375
CouncilArea      1369
Lattitude           0
Longtitude          0
Regionname          0
Propertycount       0
dtype: int64

Eksik verilen toplam sayısı: 13256
Eksik verilen oranı: 0.04648292306613367
Eksik veri bulunan satırların sayısı: 7384
Eksik veri bulunan satırların oranı: 0.543740795287187

------------------------------------------------------------------------------------  

------------------------------------------------------------------------------------  
Eksik verileri DataFrame nesnesinden silmek için DataFrame sınıfının dropna metodu 
kullanılabilir. Bu metotta default axis = 0'dır. Yani default durumda satırlar 
atılmaktadır. Ancak axis=1 parametresiyle sütunları da atabiliriz. Metot default 
durumda bize eksik verilerin atıldığı yeni bir DataFrame nesnesi vermektedir. Ancak 
metodun inplace parametresi True yapılırsa  nesne üzerinde atım yapılmaktadır. 


import pandas as pd

df = pd.read_csv('melb_data.csv')

print(f'Veri kümesinin boyutu: {df.shape}')
print("---------")

df_deleted_rows = df.dropna(axis=0)
print(f'Satır atma sonucundaki yeni boyut: {df_deleted_rows.shape}')

df_deleted_cols = df.dropna(axis=1)
print(f'Sütun atma sonucundaki yeni boyut: {df_deleted_cols.shape}')

-----------------------------------------------------------------------------------  
Eksik verilerin yerine başka değerlerin yerleştirilmesi işlemine "imputation" denilmektedir.
Kullanılan tipik imputation stratejiler şunlardır:

- Sütun sayısal ise Eksik verileri sütun ortalaması ile doldurma

- Sütun kategorik ya da sırasal ise eksik verileri mode değeri ile doldurma

- Eksik verilerin  sütunlarda uç değeler varsa medyan değeri ile doldurulması 

- Eksik verilerin yerine belli aralıkta ya da dağılımda rastgele değer yerleştirme yöntemi

- Eksik verilerin zaman serileri tarzında sütunlarda önceki ya da sonraki sütun değerleriyle doldurulması

- Eksik değerlerin regresyonla tahmin edilmesi yoluyla doldurulması

- Eksik değerlerin KNN (K-Nearest Neighbours) Yöntemi ile doldurulması

En çok uygulanan yöntem basitliği nedeniyle sütun ortalaması, sütun modu ya da 
sütun medyanı ile doldurma yöntemidir. 

-----------------------------------------------------------------------------------  
Bu veri kümesinde eksik veriler şu sütunlarda bulunmaktaydı: "Car", "BuildingArea", ,
"YearBuilt", "CouncilArea". Inputation işlemi için bu sütunların incelenmesi gerekir. 
"Car" sütunu ev için ayrılan otopark alanının kaç arabayı içerdiğini belirtmektedir. 
Bu sütunda ayrık küçük tamsayı değerler vardır. Burada imputation için sütun ortalaması 
alınabilir. Ancak bunların yuvarlanması daha uygun olabilir. Bu işlem şöyle yapılabilir:

impute_val = df['Car'].mean().round()
df['Car'] = df['Car'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)


Pandas'taki DataFrame ve Series sınıflarının mode metotları sonucu Series nesnesi 
biçiminde vermektedir. (Aynı miktarda yinelenen birden fazla değer olabileceği 
için bu değerlerin hepsinin verilmesi tercih edilmiştir.) Dolayısıyla biz bu Series 
nesnesinin ilk elemanını alarak değeri elde ettik.

impute_val = df['CouncilArea'].mode()
df['CouncilArea'] = df['CouncilArea'].fillna(impute_val[0])

-----------------------------------------------------------------------------------  
import pandas as pd

df = pd.read_csv('melb_data.csv')

print(df.isna().sum())
print("-----------------------------")

impute_val = round( df['Car'].mean())
df['Car'] = df['Car'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)

impute_val = round(df['BuildingArea'].mean())
df['BuildingArea'] = df['BuildingArea'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)

impute_val = round(df['YearBuilt'].median())
df['YearBuilt'] = df['YearBuilt'].fillna(impute_val)    # eşdeğeri df['YearBuilt'].fillna(impute_val, inplace=True)

impute_val = df['CouncilArea'].mode()
df['CouncilArea'] = df['CouncilArea'].fillna(impute_val[0])

print(f'kontrol için: {df.isna().sum()}')

-----------------------------------------------------------------------------------  
"""

# scikit-learn

"""
-----------------------------------------------------------------------------------  
NumPy ve Pandas genel amaçlı kütüphanelerdir. SciPy ise matematik ve lineer cebir 
konularına odaklanmış genel bir kütüphanedir. Oysa scikit-learn makine öğrenmesi 
amacıyla tasarlanmış ve bu amaçla kullanılan bir kütüphanedir. Kütüphanenin yüklenmesi 
şöyle yapılabilir:

Ancak scikit-learn kütüphanesi yapay sinir ağları ve derin öğrenme ağlarına yönelik 
tasarlanmamıştır. Ancak kütüphane verilerin kullanıma hazır hale getirilmesine 
ilişkin öğeleri de içermektedir. scikit-learn içerisindeki sınıflar matematiksel 
ve istatistiksel ağırlıklı öğrenme yöntemlerini uygulamaktadır. scikit-learn 
kütüphanesinin import ismi sklearn biçimindedir.

-----------------------------------------------------------------------------------  
-----------------------------------------------------------------------------------  
scikit-learn kütüphanesi genel olarak "nesne yönelimli" biçimde oluşturulmuştur. 
Yani kütüphane daha çok fonksiyonlar yoluyla değil sınıflar yoluyla kullanılmaktadır. 
Kütüphanenin belli bir kullanım biçimi vardır. Öğrenme kolay olsun diye bu biçim 
değişik sınıflarda uygulanmıştır. Kütüphanenin tipik kullanım biçimi şöyledir:

1) Önce ilgili sınıf türünden nesne yaratılır. Örneğin sınıf SimpleImputer isimli sınıf olsun:

from sklearn.impute import SimpleImputer

si = SimpleImputer(...)    

-----------------------------------------------------------------------------------  
2) Nesne yaratıldıktan sonra onun bir veri kümesi ile eğitilmesi gerekir. Buradaki 
eğitme kavramı genel amaçlı bir kavramdır. Bu işlem sınıfların fit metotlarıyla yapılmaktadır. 
fit işlemi sonrasında metot birtakım değerler elde edip onu nesnenin içerisinde saklar. 
Yani fit metotları söz konusu veri kümesini ele alarak oradan gerekli faydalı 
bilgileri elde etmektedir. Örneğin:

si.fit(dataset)

fit işlemi genel olarak transform için kullanılacak bilgilerin elde edilmesi işlemini 
yapmaktadır. DOlayısıyla fit işleminden sonra artık nesnenin özniteliklerini kullanabiliriz. 
    
-----------------------------------------------------------------------------------  
3) fit işleminden sonra fit işlemiyle elde edilen bilgilerin bir veri kümesine  
uygulanması gerekir. Bu işlem de transform metotlarıyla yapılmaktadır. Bir veri 
kümesi üzerinde fit işlemi uygulayıp birden fazla veri kümesini transform edebiliriz. 

result1 = si.transform(data1)
result2 = si.transform(data2)
...

fit ve transform metotları bizden bilgiyi NumPy dizisi olarak, Pandas, Series ya da 
DataFrame nesnesi olarak ya da Python listesi olarak alabilmektedir. fit metotları 
nesnenin kendisine geri dönmekte ve transform metotları da transform edilmiş
NumPy dizilerine geri dönmektedir.

-----------------------------------------------------------------------------------  
4) Bazen fit edilecek veri kümesi ile transform edilecek veri kümesi aynı olur. Bu 
durumda önce fit, sonra transform metotlarını çağırmak yerine bu iki işlem sınıfların 
fit_transform metotlarıyla tek hamlede de yapılabilir. Örneğin:

result = si.fit_transform(dataset)

-----------------------------------------------------------------------------------  
5) Ayrıca sınıfların fit işlemi sonucunda oluşan bilgileri almak için kullanılan 
birtakım örnek öznitelikleri ve metotları da olabilmektedir.

-----------------------------------------------------------------------------------  
6) Sınıfların bazen inverse_transform metotları da bulunmaktadır. inverse_transform 
metotları transform işleminin tersini yapmaktadır. Yani transform edilmiş bilgileri 
alıp onları transform edilmemiş hale getirmektedir. 

Genel olarak scikit-learn kütüphanesi hem NumPy hem de Pandas nesnelerini desteklemektedir. 
fit, transform ve fit_transform metotları genel olarak iki boyutlu bir veri kümesini 
kabul etmektedir. Bunun amacı genelleştirmeyi sağlamaktadır. Bu nedenle örneğin 
biz bu metotlara Pandas'ın Series nesnelerini değil DataFrame nesnelerini verebiliriz. 
-----------------------------------------------------------------------------------  
"""

"""
-----------------------------------------------------------------------------------  
# SimpleImputer 

Şimdi eksik verilerin doldurulması işleminde kolaylık sağlayan scikit-learn kütüphanesindeki 
SimpleImputer sınıfını görelim. Sınıfın kullanılması yukarıda ele alınan kalıba 
uygundur. Önce nesne yaratılır. Sonra fit işlemi yapılır. Asıl dönüştürme işlemini 
transform yapmaktadır. SimpluImputer sınıfı türünden nesne yaratılırken __init__ 
metodunda birtakım parametreler belirtilebilmektedir. Bunların çoğu zaten default 
değer almış durumdadır. Örneğin strategy parametresi default olarak 'mean' durumdadır. 
Bu impute işleminin sütunun hangi bilgisine göre yapılacağını belirtir. 'mean' 
dışında şu stratejiler bulunmaktadır:

'median'
'most_frequent'
'constant'

'constant' stratejisi belli bir değerle doldurma işlemini yapar. Eğer bu strateji 
seçilirse doldurulacak değerin fill_value parametresiyle belirtilmesi gerekir. 
Diğer parametreler için sınıfın dokümantasyonuna bakabilirsiniz

-----------------------------------------------------------------------------------  
SimpleImputer sınıfının fit, transform ve fit_transform metotları iki boyutlu bir 
dizi almaktadır. Yani bu metotlara bir NumPy dizisi geçirecekseniz onun iki boyutlu 
olması gerekir. Pandas'ın Series nesnelerinin tek boyutlu bir dizi belirttiğini 
anımsayınız. Bu durumda biz bu metotlara Series nesnesi geçemeyiz. Ancak DataFrame 
nesneleri iki boyutlu dizi belirttiği için DataFrame nesnelerini geçebiliriz. Eğer 
elimizde tek boyutlu bir dizi varsa onu bir sütundan n satırdan oluşan iki boyutlu 
bir diziye dönüştürmeliyiz. Bunun için NumPy ndarray sınıfının reshape metodunu 
kullanabilirsiniz.

----------------------------------------------------------------------------------- 
SimpleImputer nesnesi yaratılırken doldurma stratejisi nesnenin yaratımı sırasında 
verilmektedir. Yani nesne başta belirtilen stratejiyi uygulamaktadır. Ancak veri 
kümelerinin değişik sütunları değişik stratejilerle doldurulmak istenebilir. Bunun 
için birden fazla SimpleImputer nesnesi yaratılabilir. Örneğin:

si1 = SimpleImputer(strategy='mean')
si2 = SimpleImputer(strategy='median')
...

Ancak bunun yerine SimpleImputer sınıfının set_params metodu da kullanılabilir. 
Bu metot önceden belirlenmiş parametreleri değiştirmekte kullanılmaktadır. Örneğin:

si = SimpleImputer(strategy='mean')
...
si.set_params(strategy='median')
...

-----------------------------------------------------------------------------------      
SimpleImputer sınıfında yukarıda belirttiğimiz gibi fit metodu asıl doldurma işlemini 
yapmaz. Doldurma işlemi için gereken bilgileri elde eder. Yani örneğin:

a = np.array([1, 1, None, 4, None]).reshape(-1, 1)

si = SimpleImputer(strategy='mean')
si.fit(a)

Burada fit metodu aslında yalnızca bu a dizisindeki sütunların ortalamalarını elde 
etmektedir. (Örneğimizde tek br sütun var). Biz fit yaptığımız bilgiyi transform 
etmek zorunda değiliz. Örneğin:

b = np.array([1, 1, None, 4, None]).reshape(-1, 1)
result = si.transform(b)

Biz şimdi burada a'dan elde ettiğimiz ortalama 3 değeri ile bu b dizisini doldurmuş 
oluruz. Tabii genellikle bu tür durumlarda fit yapılan dizi ile transform yapılan 
dizi aynı dizi olur.

-----------------------------------------------------------------------------------  
-----------------------------------------------------------------------------------  
scikit-learn kütüphanesindeki pek çok sınıf aynı anda birden fazla sütun üzerinde 
işlem yapabilmektedir. Bu nedenle bu sınıfların fit ve transform metotları bizden 
iki boyutlu dizi istemektedir. tranform metotları da bize iki iki boyutlu dizi 
geri döndürmektedir.

Örneğin SimpleImputer sınıfına biz fit işlemind eiki bıyutlu bir dizi veriririz. 
Bu drumda fit metodu her sütunu diğerinden ayrı bir biçimde ele alır ve o sütunlara 
ilişkin bilgileri oluşturur. Örneğin biz fit metoduna aşağıdaki gibi iki boyutlu 
bir dizi vermiş olalım:

1       4
None    7
5       None
3       8
9       2

Stratejinin "mean" olduğunu varsayalım. Bu durumda fit metodu her iki sütunun da 
ortalamasını alıp nesnenin içerisinde saklayacaktır. Biz artık transform metoduna 
iki boyutlu iki sütundan oluşan bir dizi verebiliriz. Bu transform metodu bizim 
verdiğimiz dizinin ilk sütunununu fit ettiğimiz dizinin ilk sütunundan elde ettiği 
bilgiyle, ikinci sütununu fit ettiğimiz dizinin ikinci sütunundan elde ettiği bilgiyle 
dolduracakır. 

İşte sckit-learn sınıflarının fit ve transform metotlarına biz iki boyutlu diziler 
veririz. O da bize iki boyutlu diziler geri döndürür. Eğer elimizde tek boyutlu 
bir dizi varsa biz onu reshape metoduyla iki boyutlu hale getirerek fit ve transform 
metotlarına vermeliyiz.

-----------------------------------------------------------------------------------  
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')

si = SimpleImputer(strategy='mean')
df[['Car','BuildingArea']] = np.round(si.fit_transform(df[['Car','BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = si.fit_transform(df[['YearBuilt']])

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])

-----------------------------------------------------------------------------------  
scikit-learn kütüphanesinde sklearn.impute modülünde aşağıdaki imputer sınıfları 
bulunmaktadır:

SimpleImputer
IterativeImputer
MissingIndicator
KNNImputer

Buradaki KNNImputer sınıfı "en yakın komşuluk" yöntemini kullanmaktadır. Bu konu 
ileride ele alınacaktır. IterativeImputer sınıfı ise regresyon yaparak doldurulacak 
değerleri oluşturmaktadır. Örneğin biz bu sınıfa fit işleminde 5 sütunlu bir dizi 
vermiş olalım. Bu sınıf bu sütunların birini çıktı olarak dördünü girdi olarak ele 
alıp girdilerden çıktıyı tahmin edecek doğrusal bir model oluşturmaktadır. Yani bu 
sınıfta doldurulacak değerler yalnızca doldurmanın yapılacağı sütunlar dikkate 
alınarak değil diğer sütunlar da dikkate alınarak belirlenmektedir. 

Örneğin MHS veri kümesinde evin metrakaresi bilinmediğinde bu eksik veriyi ortalama 
metrakareyle doldurmak yerine "bölgeyi", "binanın yaşını" da dikkate alarak doldurmak 
isteyebiliriz. Ancak yukarıda da belirttiğimiz gibi genellikle bu tür karmaşık 
imputation işlemleri çok fazla kullanılmamaktadır. 

-----------------------------------------------------------------------------------  
"""

"""
-----------------------------------------------------------------------------------  
Verilerin kullanıma hazır hale getirilmesi sürecinin en önemli işlemlerinden biri 
de "kategorik (nominal)" ve "sıralı (ordinal)" sütunların sayısal biçime dönüştürülmesidir. 
Çünkü makine öğrenmesi algoritmaları veriler üzerinde "toplama", "çarpma" gibi 
işlemler yaparlar. Dolayısıyla kategorik veriler böylesi işlemlere sokulamazlar. 
Bunun için önce onların sayısal biçime dönüştürülmeleri gerekir. 

Genellikle bu dönüştürme "eksik verilerin ele alınması" işleminden daha sonra 
yapılmaktadır. Ancak bazen önce kategorik dönüştürmeyi yapıp sonra imputation 
işlemi de yapılabilir.

-----------------------------------------------------------------------------------  

Kategorik verilerin sayısal biçime dönüştürülmesi genellikle her kategori (sınıf) 
için 0'dan başlayarak artan bir tamsayı karşı düşürerek yapılmaktadır. Örneğin bir 
sütunda kişilerin renk tercihleri olsun. Ve sütun içeriği aşağıdaki gibi olsun:

Kırmızı
Mavi
Kırmızı
Yeşil
Mavi
Yeşil
...

Biz şimdi burada bu kategorik alanı Kırmızı = 0, Mavi = 1, Yeşil = 2 biçiminde 
sayısal hale dönüştürebiliriz. Bu durumda bu sütun şu hale gelecektir:

0
1
0
2
1
2
....

Örneğin biz bu işlemi yapan bir fonksiyon yazabiliriz. Fonksiyonun birinci parametresi 
bir DataFrame olabilir. İkinci parametresi ise hangi sütun sayısallaştıtılacağına 
ilişkin sütun isimlerini belirten dolaşılabilir bir nesne olabilir. Fonksiyonu 
şöyle yazabiliriz:

def category_encoder(df, colnames):
for colname in colnames:
    labels = df[colname].unique()
    for index, label in enumerate(labels):
        df.loc[df[colname] == label, colname] = index

Burada biz önce sütun isimlerini tek tek elde etmek için dış bir döngü kullandık. 
Sonra ilgili sütundaki "tek olan (unique)" etiketleri (labels) elde ettik. Sonra 
bu etiketleri iç bir döngüde dolaşarak sütunda ilgili etiketin bulunduğu satırlara
onları belirten sayıları yerleştirdik. 

----------------------------------------------------------------------------------- 

Test işlemi için aşağıdaki gibi "test.csv" isimli bir CSV dosyasını kullanabiliriz:
    
AdıSoyadı,Kilo,Boy,Yaş,Cinsiyet,RenkTercihi
 Sacit Bulut,78,172,34,Erkek,Kırmızı
 Ayşe Er,67,168,45,Kadın,Yeşil
 Ahmet San,85,182,32,Erkek,Kırmızı
 Macit Şen,98,192,65,Erkek,Mavi
 Talat Demir,85,181,49,Erkek,Yeşil
 Sibel Ünlü,72,172,34,Kadın,Mavi
 Ali Serçe,75,165,21,Erkek,Yeşil

Test kodu da şöyle olabilir:

    
import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')
print(df)
print('-----------------------------------------------------------------------')

def label_encode(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index
            
            
label_encode(df, ['Renk ercihi', 'Cinsiyet'])
print(df)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df.loc[df[colname] == label, colname]: Bu kısım, colname sütunundaki değerleri 
label ile eşit olan satırları seçer. df.loc[] metodunun ilk bölümü 
(df[colname] == label) hangi satırların seçileceğini belirlerken, ikinci bölüm 
(colname) hangi sütunun değerlerinin değiştirileceğini belirtir.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Şöyle bir çıktı elde edilmiştir:

        AdıSoyadı  Kilo  Boy  Yaş Cinsiyet RenkTercihi
0  Sacit Bulut    78  172   34        0            0
1      Ayşe Er    67  168   45        1            1
2    Ahmet San    85  182   32        0            0
3    Macit Şen    98  192   65        0            2
4  Talat Demir    85  181   49        0            1
5   Sibel Ünlü    72  172   34        1            2    

----------------------------------------------------------------------------------- 
Aşağıda da MHS veri kümesi üzerinde aynı işlem yapılmıştır.

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = np.round(si.fit_transform(df[['YearBuilt']]))

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])

def category_encoder(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index
    
category_encoder(df, ['Suburb', 'SellerG', 'Method', 'CouncilArea', 'Regionname'])
print(df)

----------------------------------------------------------------------------------- 
# LabelEncoder

Aslında yukarıdaki işlem scikit-learn kütüphanesindeki preprocessing modülünde 
bulunan LabelEncoder sınıfıyla yapılabilmektedir. LabelEncoder sınıfının genel 
çalışma biçimi scikit-learn kütüphanesinin diğer sınıflarındaki gibidir. Ancak bu 
sınıfın fit ve transform metotları "TEK" boyutlu bir numpy dizisi ya da Series nesnesi 
almaktadır. (Halbuki yukarıda da belirttiğimiz gibi genel olarak fit ve transform 
metotlarının çoğu iki boyutlu dizileri ya da DataFrame nesnelerini alabilmektedir.)

fit metodu yukarıda bizim yaptığımız gibi unique elemanları tespit edip bunu nesne 
içerisinde saklamaktadır. Asıl dönüştürme işlemi transform metoduyla yapılmaktadır. 
Tabii eğer fit ve transform metotlarında aynı veriler kullanılacaksa bu işlemler 
tek hamlede fit_transform metoduyla da yapılabilir. Örneğin yukarıdaki "test.csv" 
veri kümesindeki "Cinsiyet" ve "RenkTercihi" sütunlarını kategorik olmaktan çıkartıp 
sayısal biçime şöyle dönüştürebiliriz:
    
-----------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

le = LabelEncoder()

transformed_data = le.fit_transform(df['RenkTercihi'])
df['RenkTercihi'] = transformed_data

transformed_data = le.fit_transform(df['Cinsiyet'])
df['Cinsiyet'] = transformed_data
print(df)

-----------------------------------------------------------------------------------
Aşağıda MHS veri kümesindeki gerekli sütunlar LabelEncoder sınıfı ile sayısal 
biçime dönüştürülmüştür.


import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')
print(df, end='\n\n')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = np.round(si.fit_transform(df[['YearBuilt']]))

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])
    

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for colname in ['Suburb', 'SellerG', 'Method', 'CouncilArea', 'Regionname']:
    df[colname] = le.fit_transform(df[colname])

print(df)

-----------------------------------------------------------------------------------
LabelEncoder sınıfının inverse_transform metodu ters işlemi yapmaktadır. Yani bir 
kez fit işlemi yapıldıktan sonra nesne zaten hangi etiketlerin hangi sayısal değerlere 
karşı geldiğini kendi içerisinde tutmaktadır. Böylece biz sayısal değer verdiğimizde 
onun yazısal karşılığını inverse_transform ile elde edebiliriz


import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')
print(df, end='\n\n')

le = LabelEncoder()

transformed_data = le.fit_transform(df['RenkTercihi'])
df['RenkTercihi'] = transformed_data

label_names = le.inverse_transform(transformed_data)
print(label_names, end='\n\n')


transformed_data = le.fit_transform(df['Cinsiyet'])
df['Cinsiyet'] = transformed_data

label_names = le.inverse_transform(transformed_data)
print(label_names)

-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
Aslında kategorik verilerin 0'dan itibaren birer tamsayı ile numaralandırılması 
iyi bir teknik değildir. Kategorik verilen "one hot encoding" denilen biçimde 
sayısallaştırılması doğru tekniktir. Biz "one hot encoding" dönüştürmesini izleyen 
paragraflarda ele alacağız.

Sıralı (ordinal) verilerin sayısal biçime dönüştürülmesi tipik olarak düşük sıranın 
düşük numarayla ifade edilmesi biçiminde olabilir. Örneğin "Eğitim Durumu", "ilkokul", 
"ortaokul", "lise", "üniversite" biçiminde dört kategoride sıralı bir bilgi olabilir. 
Biz de bu bilgilere sırasıyla birer numara vermek isteyebiliriz. Örneğin:

İlkokul     --> 0
Ortaokul    --> 1
Lise        --> 2
Üniversite  --> 3

scikit-learn içerisindeki LabelEncoder sınıfı bu amaçla kullanılamamaktadır. Çünkü 
LabelEncoder etiketlere rastgele numara vermektedir. scikit-learn içerisinde bu 
işlemi pratik bir biçimde yapan hazır bir sınıf bulunmamaktadır. Gerçi scikit-learn 
içerisinde OrdinalEncoder isimli bir sınıf vardır ama o sınıf bu tür amaçları 
gerçekleştirmek için tasarlanmamıştır. 

-----------------------------------------------------------------------------------
OrdinalEncoder sınıfının kullanımı benzerdir. Önce fit sonra transform yapılır. 
Eğer fit ve transform metodunda aynı veri kümesi kullanılacaksa fit_transform metodu 
ile bu iki işlem bir arada yapılabilir. OrdinalEncoder sınıfının categories_ örnek 
özniteliği oluşturulan kategorileri NumPy dizisi olarak vermektedir. Sınıfın 
n_features_in_ örnek özniteliği ise fit işlemine sokulan sütunların sayısını vermektedir.


OrdinalEncoder sınıfı encode edilecek sütunları eğer onlar yazısal biçimdeyse 
"lexicographic" sıraya göre numaralandırmaktadır. (Yani sözlükte ilk gördüğü 
kategoriye düşük numara vermektedir. Tabii bu işlem UNICODE tabloya göre yapılmaktadır.) 
Kategorilere bizim istediğimiz numaraları vermemektedir. Ayrıca bu sınıfın fit ve 
transform metotları iki boyutlu nesneleri kabul etmektedir. Bu bağlamda OrdinalEncoder 
sınıfının LabelEncoder sınıfındne en önemli farklılığı OrdinalEncoder sınıfının 
birden fazla sütunu (özelliği) kabul etmesidir. Halbuki LabelEncoder sınıfı tek bir 
sütunu (özelliği) dönüştürmektedir.

-----------------------------------------------------------------------------------
Örneğin "test.csv" veri kümesi için:
    
Burada "Cinsiyet" ve "RenkTercihi" kategorik (nominal) ölçekte sütunlardır. "EğitimDurumu" 
sütunu kategorik ya da sıralı olarak ele alınabilir. Eğer biz İlkokul = 0, Ortaokul = 1, 
Lise = 2, Üniversite = 3 biçiminde sıralı ölçeğe ilişkin bir kodlama yapmak istersek 
bunu LabelEncoder ya da OrdinalEncoder ile sağlayamayız. Örneğin:

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder 

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')
print(df, end='\n\n')

oe = OrdinalEncoder()
transformed_data = oe.fit_transform(df[['Cinsiyet', 'RenkTercihi', 'Eğitim Durumu']])
df[['Cinsiyet', 'RenkTercihi', 'Eğitim Durumu']] = transformed_data 

 Buradan şöyle bir DataFrame elde edilecektir:

        AdıSoyadı  Kilo  Boy  Yaş  Cinsiyet  RenkTercihi  EğitimDurumu
0  Sacit Bulut    78  172   34       0.0           0.0            3.0
1      Ayşe Er    67  168   45       1.0           2.0            1.0
2    Ahmet San    85  182   32       0.0           0.0            3.0
3    Macit Şen    98  192   65       0.0           1.0            0.0
4  Talat Demir    85  181   49       0.0           2.0            2.0
5   Sibel Ünlü    72  172   34       1.0           1.0            1.0
6    Ali Serçe    75  165   21       0.0           2.0            3.0
                                
Bunu OridinalEncoder sınıfı le kodlamaya çalışırsak muhtemelen tam istediğimiz 
gibi bir kodlama yapamayız.   

-----------------------------------------------------------------------------------
Bu durumda eğer kategorilere istediğiniz gibi değer vermek istiyorsanız bunu manuel 
bir biçimde yapabilirsiniz. Örneğin veri kümesi read_csv fonksiyonuyla okunurken 
converters parametresi yoluyla hemen dönüştürme yapılabilir. read_csv ve load_txt 
fonksiyonlarında converters parametresi bir sözlük nesnesi almaktadır. Bu sözlük 
nesnesi hangi sütun değerleri okunurken hangi dönüştürmenin yapılacağını belirtmektedir. 
Buradaki sözlüğün her elemanının anahtarı bir sütun indeksinden ya da sütun indeksinden 
değeri ise o sütunun dönüştürülmesinde kullanılacak fonksiyondan oluşmaktadır. Tabii 
bu fonksiyon lambda ifadesi olarak da girilebilir. Örneğin:

import pandas as pd
df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv', converters={'EğitimDurumu': lambda s: {'İlkokul': 0, 'Ortaokul': 1, 'Lise': 2, 'Üniversite': 3}[s]})

Elde edilecek DataFrame nesnesi şöyle olacaktır:

        AdıSoyadı  Kilo  Boy  Yaş Cinsiyet RenkTercihi  EğitimDurumu
0  Sacit Bulut    78  172   34    Erkek      Kırmızı              0
1      Ayşe Er    67  168   45    Kadın        Yeşil              1
2    Ahmet San    85  182   32    Erkek      Kırmızı              0
3    Macit Şen    98  192   65    Erkek         Mavi              2
4  Talat Demir    85  181   49    Erkek        Yeşil              3
5   Sibel Ünlü    72  172   34    Kadın         Mavi              1
6    Ali Serçe    75  165   21    Erkek        Yeşil              0

-----------------------------------------------------------------------------------

-----------------------------------------------------------------------------------
# one hot encoding

Kategorik verilerin 0'dan itibaren LabelEncoder ya da OridnalEncoder sınıfı ile 
sayısallaştırılması iyi bir fikir değildir. Çünkü bu durumda sanki veri "sıralı (ordinal)" 
bir biçime sokulmuş gibi olur. Pek çok algoritma bu durumdan olumsuz yönde etkilenmektedir.

kategorik olguların birden fazla sütunla ifade edilmesi yoluna gidilmektedir. Kategorik 
verilerin birden fazla sütunla ifade edilmesinde en yaygın kullanılan yöntem "one hot encoding" 
denilen yöntemdir. Bu yöntemde sütundaki kategorilerin sayısı hesaplanır. Veri 
tablosuna bu sayıda sütun eklenir. "One hot" terimi "bir grup bitten hepsinin sıfır 
yalnızca bir tanesinin 1 olma durumunu" anlatmaktadır. İşte bu biçimde n tane kategori 
n tane sütunla ifade edilir. Her kategori yalnızca tek bir sütunda 1 diğer sütunlarda 
0 olacak biçimde kodlanır.

-----------------------------------------------------------------------------------

    RenkTercihi
    -----------
    Kırmızı
    Kırmızı
    Mavi
    Kırmızı
    Yeşil
    Mavi
    Yeşil
    ...

Burada 3 renk olduğunu düşünelim. Bunun "one hot encoding" dönüştürmesi şöyle olacaktır:

Kırmızı     Mavi        Yeşil
1           0           0
1           0           0
0           1           0
1           0           0
0           0           1
0           1           0
0           0           1
...

Eğer sütundaki kategori sayısı 2 tane ise böyle sütunlar üzerinde "one hot encoding" 
uygulamanın bir faydası yoktur. Bu tür ikili sütunlar 0 ve 1 biçiminde kodlanabilir. 
(Yani bu işlem LabelEncoder sınıfyla yapılabilir).

-----------------------------------------------------------------------------------
"One hot encoding" yapmanın çok çeşitli yöntemleri vardır. Örneğin scitkit-learn 
içerisindeki preprocessing modülünde bulunan OneHotEncoder sınıfı bu işlem için 
kullanılabilir. Sınıfın genel kullanımı diğer sckit-learn sınıflarında olduğu gibidir. 
Yani önce fit işlemi sonra transform işlemi yapılır. fit işleminden sonra sütunlardaki 
"tek olan (unique)" elemanlar sınıfın categories_ örnek özniteliğine NumPy dizilerinden 
oluşan bir liste biçiminde kaydedilmektedir.

"One hot encoding" yapılırken DataFrame içerisine eski sütunu silip yeni sütunlar 
eklenmelidir. DataFrame sütununa bir isim vererek atama yapılırsa nesne o sütunu 
zaten otomatik eklemektedir. "One hot encoding" ile oluşturulan sütunların isimleri 
önemli değildir. Ancak OneHotEncoder nesnesi önce sütunu np.unique fonksiyonuna 
sokmakta ondan sonra o sırada encoding işlemi yapmaktadır. NumPy'ın unique fonksiyonu 
aynı zamanda sıraya dizme işlemini de zaten yapmaktadır. Dolayısıyla OneHotEncoder 
aslında kategorik değerleri alfabetik sıraya göre sütunsal olarak dönüştürmektedir.

OneHotEncoder sınıfının fit ve transform metotları çok boyutlu dizileri kabul 
etmektedir. Bu durumda biz bu metotlara Pandas'ın Series nesnesini değil DataFrame 
nesnesini vermeliyiz.

OneHotEncoder nesnesini yaratırken "sparse_output" parametresini False biçimde vermeyi 
unutmayınız. (Bu parametrenin eski ismi yalnızca "sparse" biçimindeydi). Çünkü bu 
sınıf default olarak transform edilmiş nesneyi "seyrek matris (sparse matrix)" 
olarak vermektedir. Elemanlarının büyük çoğunluğu 0 olan matrislere "seyrek matris 
(sparse matrix)" denilmektedir. Bu tür matrisler "fazla yer kaplamasın diye" 
sıkıştırılmış bir biçimde ifade edilebilmektedir. 

Yine OneHotEncoder nesnesi yaratılırken parametre olarak dtype türünü belirtebiliriz. 
Default dtype türü np.float64 alınmaktadır. Matris seyrek formda elde edilmeyecekse 
bu dtype türünü "uint8" gibi en küçük türde tutabilirsiniz. max_categories parametresi 
kategori sayısını belli bir değerde tutmak için kullanılmaktadır. Bu durumda diğer 
tüm kategoriler başka bir kategori oluşturmaktadır

-----------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

ohe = OneHotEncoder(sparse=False, dtype='uint8')
transformed_data = ohe.fit_transform(df[['RenkTercihi']])

df.drop(['RenkTercihi'], axis=1, inplace=True)

df[ohe.categories_[0]] = transformed_data

# ohe.categories_  -----> [ array(['Kırmızı', 'Mavi', 'Yeşil'], dtype=object) ]
print(df)

-----------------------------------------------------------------------------------
DataFrame nesnesine yukarıdaki gibi birden fazla sütun eklerken dikkat etmek gerekir. 
Çünkü tesadüfen bu kategori isimlerine ilişkin sütunlardan biri zaten varsa o sütun 
yok edilip yerine bu kategori sütunu oluşturulacaktır. Bunu engellemek için oluşturacağınız 
kategori sütunlarını önek vererek isimlendirebilirsiniz. Önek verirken orijinal 
sütun ismini kullanırsanız bu durumda çakışma olmayacağı garanti edilebilir. Yani 
örneğin RenkTercihi sütunu için "Kırmızı", "Mavi" "Yeşil" isimleri yerine 
"RenkTercihi_Kırmızı", "RenkTercihi_Mavi" ve "RenkTercihi_Yeşil" isimlerini 
kullanabilirsiniz. Bu biçimde isim elde etmek "liste içlemiyle" oldukça kolaydır. 
Örneğin:

category_names = ['RenkTercihi_' + category for category in ohe.categories_[0]]


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

ohe = OneHotEncoder(sparse=False, dtype='uint8')

transformed_data = ohe.fit_transform(df[['RenkTercihi']])

df.drop(['RenkTercihi'], axis=1, inplace=True)

category_names = ['RenkTercihi_' + category for category in ohe.categories_[0]]

df[category_names] = transformed_data

print(df)
    
-----------------------------------------------------------------------------------
Burada "RenkTercihi"nin yanı sıra "Eğitim Durumu" de kategorik bir sütundur. Bunun her 
ikisini birden tek hamlede "one hot encoding" işlemine sokabiliriz:

ohe = OneHotEncoder(sparse=False, dtype='uint8')
transformed_data = ohe.fit_transform(df[['RenkTercihi', 'Eğitim Durumu']])

df.drop(['RenkTercihi', 'Eğitim Durumu'], axis=1, inplace=True)

categories1 = ['RenkTercihi_' + category for category in ohe.categories_[0]]
categories2 = ['Eğitim Durumu_' + category for category in ohe.categories_[1]]

df[categories1 + categories2] = transformed_data

----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
# get_dummies

One hot encoding yapmanın diğer bir yolu Pandas kütüphanesindeki get_dummies fonksiyonunu 
kullanmaktadır. get_dummies fonksiyonu bizden bir DataFrame, Series ya da dolaşılabilir 
herhangi bir nesneyi alır. Eğer biz get_dummies fonksiyonuna bütün bir DataFrame
geçirirsek fonksiyon oldukça akıllı davranmaktadır. Bu durumda fonksiyon DataFrame 
nesnesi içerisindeki yazısal sütunları tespit eder. Yalnızca yazısal sütunları 
"one hot encoding" işlemine sokar ve bize yazısal sütunları dönüştürülmüş yeni bir 
DataFrame nesnesi verir. Pandas ile çalışırken bu fonksiyon çok kolaylık sağlamaktadır.

Biz aslında get_dummies fonksiyonu yoluyla yapmış olduğumuz işlemleri tek hamlede 
yapabiliriz:

    
import pandas as pd
df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')    

transformed_df = pd.get_dummies(df, dtype='uint8')


Burada biz tek hamlede istediğimiz dönüştürmeyi yapabildik. Bu dönüştürmede yine 
sütun isimleri orijinal sütun isimleri ve kategori isimleriyle örneklendirilmiştir. 
Eğer isterse programcı "prefix" parametresi ile bu öneki değiştirebilir, "prefix_sep"
parametresiyle de '_' karakteri yerine başka birleştirme karakterlerini kullanabilir. 

transformed_df = pd.get_dummies(df, columns=['RenkTercihi', 'AdıSoyadı'], dtype='uint8', prefix=['R', 'AD'], prefix_sep='-')

get_dummies fonksiyonu default durumda sparse olmayan bool türden bir DataFrame 
nesnesi vermektedir. Ancak get_dummies fonksiyonunda "dtype" parametresi belirtilerek 
"uint8" gibi bir türden çıktı oluşturulması sağlanabilmektedir. 

----------------------------------------------------------------------------------
Biz bir DataFrame nesnesinin tüm yazısal sütunlarını değil bazı yazısal sütunlarını 
da "one hot encoding" işlemine sokmak isteyebiliriz. Bu durumda fonksiyon DataFrame 
nesnesinin diğer sütunlarına hiç dokunmamaktadır. Örneğin:

transformed_df = pd.get_dummies(df, columns=['RenkTercihi'], dtype='uint8')


Biz burada yalnızca DataFrame nesnesinin "RenkTercihi" sütununu "one hot encoding" 
yapmış olduk. get_dummies fonksiyonun zaten "one hot encoding" yapılan sütunu 
sildiğine dikkat ediniz. Bu bizim genellikle istediğimiz bir şeydir. Yukarıdaki 
örnekte "test.csv" dosyasında "AdıSoyadı" sütunu yazısal bir sütundur. Dolayısıyla 
default durumda bu sütun da "one hot encoding" işlemine sokulacaktır. Bunu engellemek 
için "columns" parametresinden faydalanabiliriz ya da baştan o sütunu atabiliriz. 
Örneğin:

transformed_df = pd.get_dummies(df.iloc[:, 1:], dtype='uint8')

----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
# to_categorical

Diğer bir "one hot encoding" uygulama yöntemi de "tensorflow.keras" kütüphanesindeki 
"to_categorical" fonksiyonudur. Bazen zaten Keras ile çalışıyorsak bu fonksiyonu 
tercih edebilmekteyiz. to_categorical fonksiyonunu kullanmadan önce kategorik sütunun 
sayısal biçime dönüştürülmüş olması gerekmektedir. Yani biz önce sütun üzerinde 
eğer sütun yazısal ise LabelEncoder işlemini uygulamalıız. to_categorical fonksiyonu 
aynı anda birden fazla sütunu "one hot encoding" yapamamaktadır. Bu nedenle diğer 
seçeneklere göre kullanımı daha zor bir fonksiyondur. to_categorical fonksiyonu 
Keras kütüphanesindeki utils isimli modülde bulunmaktadır.
----------------------------------------------------------------------------------

import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv') 

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

transformed_color = le.fit_transform(df['RenkTercihi'])
transformed_occupation = le.fit_transform(df['Meslek'])

ohe_color = to_categorical(transformed_color)
ohe_occupation = to_categorical(transformed_occupation)

color_categories = ['RenkTercihi_' + color for color in df['RenkTercihi'].unique()]
occupation_categories = ['Meslek_' + occupation for occupation in df['Meslek'].unique()]

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)

df[color_categories] = ohe_color
df[occupation_categories] = ohe_occupation

print(df)

----------------------------------------------------------------------------------
"One hot encoding" yapmanın diğer bir yolu da manuel yoldur.
 
 
import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

print(df, end='\n\n')

import numpy as np

color_cats = np.unique(df['RenkTercihi'].to_numpy())
occupation_cats = np.unique(df['Meslek'].to_numpy())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['RenkTercihi'] = le.fit_transform(df['RenkTercihi'])
df['Meslek'] = le.fit_transform(df['Meslek'])

print(df, end='\n\n')

color_um = np.eye(len(color_cats))
occupation_um = np.eye(len(occupation_cats))

ohe_color = color_um[df['RenkTercihi'].to_numpy()]
ohe_occupation = occupation_um[df['Meslek'].to_numpy()]

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)
df[color_cats] = ohe_color
df[occupation_cats] = ohe_occupation

print(df, end='\n\n')

----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
# dummy variable encoding

"One hot encoding" işleminin bir versiyonuna da "dummy variable encoding" denilmektedir. 
Şöyle ki: "One hot encoding" işleminden tane kategori için n tane sütun oluşturuluyordu. 
Halbuki "dummy variable encoding" işleminde n tane kategori için n - 1 tane sütun 
oluşturulmaktadır. Çünkü bu yöntemde bir kategori tüm sütunlardaki sayının 0 olması 
ile ifade edilmektedir. Örneğin Kırmızı, Yeşil, Mavi kategorilerinin bulunduğu 
bir sütun şöyle "dummy variable encoding" biçiminde dönüştürülebilir:

Mavi Yeşil
0       0       (Kırmızı)
1       0       (Mavi)
0       1       (Yeşil)

Görüldüğü gibi kategorilerden biri (burada "Kırmızı") tüm elemanı 0 olan satırla 
temsil edilmiştir. Böylece sütun sayısı bir eksiltilmiştir.

---------------------------------------------------------------------------------
"Dummy variable encoding" işlemi için farklı sınıflar ya da fonksiyonlar kullanılmamaktadır. 
Bu işlem "one hot encoding" yapan sınıflar ve fonksiyonlarda özel bir parametreyle 
gerçekleştirilmektedir. Örneğin scikit-learn kütüphanesindeki OneHotEncoder 
sınıfının drop parametresi 'first' olarak geçilirse bu durumda transform işlemi 
"dummy variable encoding" biçiminde yapılmaktadır.

---------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

ohe = OneHotEncoder(sparse=False, drop='first')
transformed_data = ohe.fit_transform(df[['RenkTercihi']]) 

print(df['RenkTercihi'])
print()
print(ohe.categories_)
print()
print(transformed_data)

Görüldüğü gibi burada "Kırmızı" kategorisi [0, 0] biçiminde kodlanmıştır. 

---------------------------------------------------------------------------------
Pandas'ın get_dummies fonksiyonunda drop_first parametresi True geçilirse 
"dummy variable encoding" uygulanmaktadır. Örneğin:

    
transformed_df = pd.get_dummies(df, columns=['RenkTercihi', 'Meslek'], dtype='uint8', drop_first=True)
print(transformed_df)

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
# Binary encoding 

Binary encoding yönteminde her kategori "ikilik sistemde bir sayıymış" gibi ifade 
edilmektedir. Örneğin sütunda 256 tane kategori olsun. Bu kategoriler 0'dan 255'e 
kadar numaralandırılabilir. 0 ile 255 arasındaki sayılar 2'lik sistemde 8 bit ile
ifade edilebilir. Örneğin bir kategorinin sayısal değeri (LabelEncoder yapıldığını 
düşünelim) 14 olsun. Biz bu kategoriyi aşağıdaki gibi 8 bit'lik 2'lik sistemde 
bir sayı biçiminde kodlayabiliriz:

0 0 0 0 1 1 1 0 

Tabii kategori sayısı tam 2'nin kuvveti kadar olmak zorunda değildir. Bu durumda 
kategori sayısı N olmak üzere gerkeli olan bit sayısı (yani sütun sayısı) 
ceil(log2(N)) hesabı ile elde edilebilir. 

!! ceil fonksiyonu float sayıyı üste yuvarlar

scikit-learn kütüphanesinin contribute girişimlerinden birinde bu işlemi yapan bir 
BinaryEncoder isminde bir sınıf bulunmaktadır. Bu sınıf category_encoders isimli 
bir paket içerisindedir ve bu paket ayrıca yüklenmelidir. Yükleme şöyle yapılabilir:

pip install category_encoders

---------------------------------------------------------------------------------
BinaryEncoder sınıfının transform fonksiyonu default durumda Pandas DataFrame nesnesi 
vermektedir. Ancak nesne yaratılırken return_df parametresi False geçilirse bu 
durumda transform fonksiyonları NumPy dizisi geri döndürmektedir.

from category_encoders.binary import BinaryEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

be = BinaryEncoder()
transformed_data = be.fit_transform(df['Meslek'])
print(transformed_data) 

Burada "Meslek" sütunu "binary encoding" biçiminde kodlanmıştır. BinaryEncode kodlamada değerleri 1'den başlatılmaktadır. 
Yukarıdaki işlemden aşağıdaki gibi bir çıktı elde edilmiştir:

   Meslek_0  Meslek_1
0         0         1
1         0         1
2         1         0
3         1         1
4         1         0
5         1         1
6         0         1

---------------------------------------------------------------------------------
import pandas as pd
  
df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

from category_encoders.binary import BinaryEncoder

be = BinaryEncoder()
transformed_data = be.fit_transform(df[['RenkTercihi', 'Meslek']])
print(transformed_data)

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)
df[transformed_data.columns] = transformed_data    # pd.concat((df, transformed_data), axis=1)
print(df)

---------------------------------------------------------------------------------
Tıpkı get_dummies fonksiyonunda olduğu gibi aslında bir DataFrame bütünsel olarak 
da verilebilir. Yine default durumda tüm yazısal sütunlar "binary encoding" 
dönüştürmesine sokulmaktadır. Ancak biz BinaryEncoding sınıfının __init__ metodunda
cols parametresi ile hangi sütunların dönüştürüleceğini belirleyebiliriz.


import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

from category_encoders.binary import BinaryEncoder

be = BinaryEncoder(cols=['RenkTercihi', 'Meslek'])
transformed_df = be.fit_transform(df)
print(transformed_df)

---------------------------------------------------------------------------------
"""



#  -------------------  Yapay Sinir Ağları (Artificial Neural Neetworks)  -------------------

"""
---------------------------------------------------------------------------------
Yapay zeka ve makine öğrenmesi alanının en önemli yöntemlerinin başında "yapay 
sinir ağları (artificial neural networks)" ve "derin öğrenme (deep learning)" denilen 
yöntemler gelmektedir. Biz de bu bölümde belli bir derinlikte bu konuları ele alacağız. 

Yapay sinir ağları yapay nöronların birbirlerine bağlanmasıyla oluşturulmaktadır. 
Bir nöronun girdileri vardır ve yalnızca bir tane de çıktısı vardır. Nöronun girdileri 
veri kümesindeki satırları temsil eder. Yani veri kümesindeki satırlar nöronun 
girdileri olarak kullanılmaktadır. Nöronun girdilerini xi temsil edersek her girdi 
"ağırlık (weight) değeri" denilen bir değerler çarpılır ve bu çarpımların toplamları 
elde edilir. Ağırlık değerlerini wi ile gösteribiliriz. Bu durumda xi'lerle wi'ler 
karşılıklı olarak çarpılıp toplanmaktadır.

total = x1w1 + x2w2 + x3w3 + x4w4 + x5w5

İki vektörün karşılıklı elemanlarının çarpımlarının toplamına İngilizce "dot product" 
denilmektedir. (Dot product işlemi np.dot fonksiyonuyla yapılabilmektedir.) Elde 
edilen dot product "bias" denilen bir değerle toplanır. Biz bias dieğerini b ile 
temsil edeceğiz. Örneğin:

total = x1w1 + x2w2 + x3w3 + x4w4 + x5w5 + b

Bu toplam da "aktivasyon fonsiyonu (activation function)" ya da "transfer fonksiyonu 
(transfer function)" denilen bir fonksiyona sokulmaktadır. Böylece o nöronun çıktısı 
elde edilmektedir. Örneğin:

out = activation(x1w1 + x2w2 + x3w3 + x4w4 + x5w5 + b)

Biz bu işlemi vektörel olarak şöyle de gösterebiliriz:

out = activation(XW + b)

---------------------------------------------------------------------------------
Bir nöronun çıktısı başka nöronlar girdi yapılabilir. Böylece tüm ağın nihai çıktıları 
oluşur. Örneğin ağımızın bir katmanında k tane nöron olsun. Tüm girdilerin 
(yani Xi'lerin) bu nöronların hepsine bağlandığını varsayalım. Her nöronun ağırlık 
değerleri ve bias değeri diğerlerinden farklıdır.

out = activation(XW + b)

X = [x1, x2, x3, ..., xn]

W matrisi aşağıdaki görünümdedir:

w11  w21 w31 ... wk1
w12  w22 w31 ... wk2
w13  w23 w33 ... wk3
...  ... ... ... ...
w1n  w2n w3n ... wkn


Buradaki X matrisi ile W matrisi matrisi, matris çarpımına sokulduğunda 1XK boyutunda 
bir matris elde edilecektir. Gösterimimizdeki b matrisi şöyle temsil edilebilir:

b = [b1, b2, b2, ...., bk]

Böylece XW + b işleminden 1XK boyutunda bir matris elde edilecektir. 

---------------------------------------------------------------------------------
Bir sinir ağının amacı bir "kestirimde" bulunmaktır. Yani biz ağa girdi olarak Xi 
değerlerini veririz. Ağdan hedeflediğimiz çıktıyı elde etmeye çalışırız. Ancak 
bunu yapabilmemiz için nöronlardaki w değerlerinin ve b değerlerinin biliniyor 
olması gerekir. Önce biz ağımızı mevcut verilerle eğitip bu w ve b değerlerinin 
uygun biçimde oluşturulmasını sağlarız. Ondan sonra kestirim yaparız. Tabi ağ ne 
kadar iyi eğitilirse ağın yapacağı kestirim de o kadar isabetli olacaktır. 

---------------------------------------------------------------------------------
İstatistikte girdi değerlerinden hareketle çıktı değerinin belirlenmesine (tahmin 
edilmesine) yönelik süreçlere "regresyon (regression)" denilmektedir. Regresyon 
işlemleri çok çeşitli biçimlerde sınıflandırılabilmektedir. Çıktıya göre regresyon 
işlemleri istatistikte tipik olarak iki grupla sınıflandırılmaktadır:

1) (Lojistik Olmayan) Regresyon İşlemleri
2) Lojistik Regresyon İşlemleri

Aslında istatistikte "regresyon" denildiğinde çoğu kez default olarak zaten 
"lojistik olmayan regresyon" işlemleri anlaşılmaktadır. Bu tür regresyonlarda 
girdilerden hareketle kategorik değil sayısal bir çıktı elde edilmektedir.

Girdiler bir resmin pixel'leri olabilir. Çıktı da bu resmin elma mı, armut mu, 
kayısı mı olduğuna yönelik kategorik bir bilgi olabilir. Bu tür regresyonlara 
istatistikte "lojistik regresyonlar" ya da "logit regresyonları" denilmektedir.

Makine öğrenmesinde lojistik regresyon terimi yerine daha çok "sınıflandırma 
(classification)" terimi kullanılmaktadır. Buradaki "lojistik" sözcüğü mantıktaki 
"lojikten" gelmektedir. İngilizce "logistic" terimi "logic" ve "statistics" 
terimlerinin birleştirilmesiyle uydurulmuştur.

Lojistik olsun ya da olmasın aslında regresyon işlemlerinin hepsi girdiyi çıktıya 
dönüştüren bir f foonksiyonunun elde edilmesi sürecidir. Örneğin:

y = f(x1, x2, ..., xn)

İşte makine öğrenmesinde bu f fonksiyonunun elde edilmesinin çeşitli yöntemleri 
vardır. Yapay sinir ağlarında da aslında bu biçimde bir f fonksiyonu bulunmaya 
çalışılmaktadır.

---------------------------------------------------------------------------------
Sınıflandırma sürecindeki çıktının olabileceği değerlere "sınıf (class)" denilmektedir.  
Sınıflandırma (lojistik regresyon) problemlerinde eğer çıktı ancak iki değerden 
biri olabiliyorsa bu tür sınıflandırma problemlerine de "iki sınıflı sınıflandırma 
(binary classification)" problemleri denilmektedir. Örneğin bir film hakkında 
yazılan yorum yazısının "olumlu" ya da "olumsuz" biçiminde iki değerden oluştuğunu 
düşünelim. Bu sınıflandırma işlemi ikili sınıflandırmadır. Benzer biçimde bir 
biyomedikal görüntüdeki kitlenin "iyi huylu (benign)" mu "kötü huylu (malign)" mu 
olduğuna yönelik sınıflandırma da ikili sınıflandırmaya örnektir. 

Eğer sınıflandırmada çıktı sınıflarının sayısı ikiden fazla ise böyle sınıflandırma 
problemlerine "çok sınıflı (multiclass)" sınıflandırma problemleri denilmektedir. 
Örneğin bir resmin hangi meyveye ilişkin olduğunun tespit edilmesi için kullanılan 
sınıflandırma modeli "çok sınıflı" bir modeldir. 

İstatistikte "lojistik regresyon" denildiğinde aslında default olarak "iki sınıflı 
(binary)" lojistik regresyon anlaşılmaktadır. Çok sınıflı lojistik regresyonlara 
istatistikte genellikle İngilizce "multiclass logistic regression" ya da 
"multinomial logistic regression" denilmektedir.

---------------------------------------------------------------------------------
En basit yapay sinir ağı mimarisi tek bir nörondan oluşan mimaridir. Buna 
"perceptron" denilmektedir. Aşağıda bir nöronun bir sınıfla temsil edilmesine ilişkin 
bir örnek 

import numpy as np

class Neuron:
    def __init__(self, w, b):
        self.w= w
        self.b = b
        
    def output(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)
    
    @staticmethod
    def sigmoid(x):
        return np.e ** x / (1 + np.e ** x)

        
w = np.array([1, 2, 3])
b = 1.2

n = Neuron(w, b)

x = np.array([1, 3, 4])
result = n.output(x)
print(result)
---------------------------------------------------------------------------------
"""


#  Yapay Sinir Ağlarında Katmanlar (LAYER)

"""
---------------------------------------------------------------------------------
Bir yapay sinir ağı modelinde "katmanlar (layers)" vardır. Katman aynı düzeydeki 
nöron grubuna denilmektedir. Yapya sinir ağı katmanları tipik olarak üçe ayırmaktadır:

1) Girdi Katmanı (Input Layer)
2) Saklı Katmanlar (Hidden Layers)
3) Çıktı Katmanı (Output Layer)

Girdi katmanı veri kümesindeki satırları temsil eden yani ağa uygulanacak verileri 
belirten katmandır. Aslında girdi katmanı gerçek anlamda nöronlardan oluşmaz. Ancak 
anlatımları kolaylaştırmak için bu katmanın da nöronlardan oluştuğu varsayılmaktadır. 
Başka bir deyişle girdi katmanının tek bir nöron girişi ve tek bir çıktısı vardır.
Yani girdi katmanı bir şey yapmaz, girdiyi değiştirmeden çıktıya verir. Girdi 
katmanındaki nöron sayısı veri kümesindeki sütunların (yani özelliklerin) sayısı 
kadar olmalıdır. Örneğin 5 tane girdiye (özelliğe) sahip olan bir sinir ağının 
girdi katmanı şöyle gösterilebilir:

x1 ---> O --->
x2 ---> O --->
x3 ---> O --->
x4 ---> O --->
x5 ---> O --->

Buradaki O sembolleri girdi katmanındaki nöronları temsil etmektedir. Girdi katmanındaki 
nöronların 1 tane girdisinin 1 tane de çıktısınn olduğuna dikkat ediniz. Buradaki 
nöronlar girdiyi değiştirmediğine göre bunların w değerleri 1, b değerleri 0, 
aktivasyon fonksiyonu da f(x) = x biçiminde olmalıdır.


Hidden Layer -> Girdiler saklı katman denilen katmanlardaki nöronlara bağlanırlar. 
Modelde sıfır tane, bir tane ya da birden fazla saklı katman bulunabilir. Saklı 
katmanların sayısı ve saklı katmanlardaki nöronların sayısı ve bağlantı biçimleri 
problemin niteliğine göre değişebilmektedir. Yani saklı katmanlardaki nöronların 
girdi katmanıyla aynı sayıda olması gerekmez. Her saklı katmandaki nöron sayıları 
da aynı olmak zorunda değildir.


Çıktı katmanı bizim sonucu alacağımız katmandır. Çıktı katmanındaki nöron sayısı 
bizim kestirmeye çalıştığımız olgularla ilgilidir. Örneğin biz bir evin fiyatını 
kestirmeye çalışıyorsak çıktı katmanında tek bir nöron bulunur. Yine örneğin biz 
ikili sınıflandırma problemi üzerinde çalışıyorsak çıktı katmanı yine tek bir nörondan 
oluşabilir. Ancak biz evin fiyatının yanı sıra evin sağlamlığını da kestirmek 
istiyorsak bu durumda çıktı katmanında iki nöron olacaktır. Benzer biçimde çok 
sınıflı sınıflandırma problemlerinde çıktı katmanında sınıf sayısı kadar nöron bulunur.

---------------------------------------------------------------------------------
Bir yapay sinir ağı modelinde katman sayısının artırılması daha iyi bir sonucun 
elde edileceği anlamına gelmez. Benzer biçimde katmanlardaki nöron sayılarının 
artırılması da daha iyi bir sonucun elde edileceği anlamına gelmemektedir. Katmanların 
sayısından ziyade onların işlevleri daha önemli olmaktadır. Ağa gereksiz katman 
eklemek, katmanlardaki nöronları artırmak tam ters bir biçimde ağın başarısının 
düşmesine de yol açabilmektedir. Yani gerekmediği halde ağa saklı katman eklemek, 
katmanlardaki nöron sayısını artırmak bir fayda sağlamamakta tersine kestirim 
başarısını düşürebilmektedir. Ancak görüntü tanıma gibi özel ve zor problemlerde 
saklı katman sayılarının artırılması gerekebilmektedir. 

---------------------------------------------------------------------------------
Pekiyi bir sinir ağı modelinde kaç tane saklı katman olmalıdır? Pratik olarak 
şunları söyleyebiliriz:
    
- Sıfır tane saklı katmana sahip tek bir nörondan oluşan en basit modele "perceptron" 
dendiğini belirtmiştir. Bu perceptron "doğrusal olarak ayrıştırılabilen (linearly 
separable)" sınıflandırma problemlerini ve yalın doğrusal regresyon problemlerini 
çözebilmektedir. 

- Tek saklı katmanlı modeller aslında pek çok sınıflandırma problemini ve (doğrusal 
olmayan) regresyon problemlerini belli bir yeterlilikte çözebilmektedir. Ancak 
tek saklı katman yine de bu tarz bazı problemler için yetersiz kalabilmektedir. 

- İki saklı katman pek çok karmaşık olmayan sınıflandırma problemi için ve regresyon 
problemi için iyi bir modeldir. Bu nedenle karmaşık olmayan problemler için ilk 
akla gelecek model iki saklı katmanlı modeldir. 

- İkiden fazla saklı katmana sahip olan modeller karmaşık ve özel problemleri çözmek 
için kullanılmaktadır. İki saklı katmandan fazla katmana sahip olan modellere genel 
olarak "derin öğrenme ağları (deep learning networks)" denilmektedir. 

Yukarıda da belirttiğimiz gibi "derin öğrenme (deep learning)" farklı bir yöntemi 
belirtmemektedir. Derin öğrenme özel ve karmaşık problemleri çözebilmek için ikiden 
fazla saklı katman içeren sinir ağı modellerini belirtmek için kullanılan bir 
terimdir.    

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Bir veri kümesini CSV dosyasından okuduktan sonra onu Keras'ın kullanımına hazırlamak 
için bazı işlemlerin yapılması gerekir. Yapılması gereken ilk işlem veri kümesinin 
dataset_x ve dataset_y biçiminde iki parçaya ayrılmasıdır. Çünkü ağın eğitilmesi,
sırasında girdilerle çıktıların ayrıştırılması gerekmektedir. Burada dataset_x 
girdileri dataset_y ise kestirilecek çıktıları belirtmektedir. 

Eğitim bittikten sonra genellikle ağın kestirimine hangi ölçüde güvenileceğini 
belirleyebilmek için bir test işlemi yapılır. Ağın kestirim başarısı "test veri kümesi" 
denilen bir veri kümesi ile yapılmaktadır. Test veri kümesinin eğitimde kullanılmayan 
bir veri kümesi biçiminde olması gerekir.

Eğitim ve test veri kümesini manuel olarak ayırabiliriz. Ancak ayırma işleminden 
önce veri kümesini satırsal bakımdan karıştırmak uygun olur. Çünkü bazı veri kümeleri 
CSV dosyasına karışık bir biçimde değil yanlı bir biçimde kodlanmış olabilmektedir. 
Örneğin bazı veri kümeleri bazı alanlara göre sıraya dizilmiş bir biçimde bulunabilmektedir. 
Biz onun baştaki belli kısmını eğitim, sondaki belli kısmını test veri kümesi 
olarak kullanırsak eğitim ve test veri kümeleri yanlı hale gelebilmektedir.
-----> np.random.shuffle(dataset)

---------------------------------------------------------------------------------
diabetes.csv bazı sütunlar eksik veri içermektedir. Bu eksik verile NaN biçiminde 
değil 0 biçiminde kodlanmıştır. Biz bu eksik verileri ortalama değerle doldurabiliriz. 
Eksik veri içeren sütunlar şunlardır:

Glucose
BloodPressure
SkinThickness
Insulin
BMI

---------------------------------------------------------------------------------
TRAINING_RATIO = 0.80

import pandas as pd
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\2- KerasIntroduction\diabetes.csv')


from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='mean', missing_values=0 )

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])
print((df==0).sum())
print()

dataset = df.to_numpy()

import numpy as np

np.random.shuffle(dataset)

dataset_x = dataset[:, :-1] 
dataset_y = dataset[:, -1] 

training_len = int(np.round(len(dataset) * TRAINING_RATIO))

training_dataset_x = dataset_x[ :training_len]
test_dataset_x = dataset_x[training_len : ]

training_dataset_y = dataset_y[ :training_len]
test_dataset_y = dataset_y[training_len : ]

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Veri kümesini eğitim ve test olarak ayırma işlemi için sklearn.model_selection 
modülündeki train_test_split isimli fonksiyon sıkça kullanılmaktadır. Fonksiyon 
NumPy dizilerini ya da Pandas DataFrame ve Series nesnelerini ayırabilmektedir. 

Fonksiyon bizden dataset_x ve dataset_y değerlerini ayrı ayrı ister. test_size 
ya da train_size parametreleri 0 ile 1 arasında test ya da eğitim verilerinin oranını 
belirlemek için kullanılmaktadır. train_test_split fonksiyonu bize 4'lü bir liste 
vermektedir. Listenin elemanları sırasıyla şunlardır: training_dataset_x, 
test_dataset_x, training_dataset_y, test_dataset_y. Örneğin:

    
from sklearn.model_selection import train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

test_size = 0.2  ---> TRAINING_RATIO = 0.8 ---> training_size = 0.8

Burada fonksiyona dataset_x ve dataset_y girdi olarak verilmiştir. Fonksiyon 
bunları bölerek dörtlü bir listeye geri dönmüştür.

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Keras'ta bir sinir ağı oluşturmanın çeşitli adımları vardır. Burada sırasıyla bu 
adımlardan ve adımlarla ilgili bazı olgulardan bahsedeceğiz.


1-) Öncelikle bir model nesnesi oluşturulmalıdır. tensorflow.keras modülü içerisinde 
çeşitli model sınıfları bulunmaktadır. En çok kullanılan model sınıfı Sequential 
isimli sınıftır. Tüm model sınıfları Model isimli sınıftan türetilmiştir. Sequential 
modelde ağa her eklenen katman sona eklenir. Böylece ağ katmanların sırasıyla 
eklenmesiyle oluşturulur. Sequential nesnesi yaratılırken name parametresiyle modele 
bir isim de verilebilir. Örneğin:

from tensorflow.keras import Sequential

model = Sequential(name='Sample')

Aslında Sequential nesnesi yaratılırken katmanlar belirlendiyse layers parametresiyle 
bu katmanlar da verilebilmektedir. Ancak sınıfın tipik kullanımında katmanlar daha 
sonra izleyen maddelerde ele alınacağı gibi sırasıyla eklenmektedir.

---------------------------------------------------------------------------------

2-) Model nesnesinin yaratılmasından sonra katman nesnelerinin oluşturulup model nesnesine 
eklenmesi gerekir. Keras'ta farklı gereksinimler için farklı katman sınıfları 
bulundurulmuştur. En çok kullanılan katman sınıfı tensorflow.keras.layers modülündeki 
Dense sınıfıdır. Dense bir katman modele eklendiğinde önceki katmandaki tüm nöronların 
çıktıları eklenen katmandaki nöronların hepsine girdi yapılmaktadır. 

Bu durumda örneğin önceki katmanda k tane nöron varsa biz de modele n tane nörondan 
oluşan bir Dense katman ekliyorsak bu durumda modele k * n + n tane yeni parametre 
(yani tahmin edilmesi gereken parametre) eklemiş oluruz. Burada k * n tane ayarlanması 
gereken w (ağırlık) değerleri ve n tane de ayarlanması gereken bias değerleri söz 
konusudur. 

!!! Bir nörondaki w (ağırlık) değerlerinin o nörona giren nöron sayısı kadar 
olduğuna ve bias değerlerinin her nöron için bir tane olduğuna dikkat ediniz. !!!

Dense sınıfının __init__ metodunun ilk parametresi eklenecek katmandaki nöron sayısını 
belirtir. İkinci parametre olan activation parametresi o katmandaki tüm nöronların 
aktivasyon fonksiyonlarının ne olacağını belirtmektedir. Bu parametreye aktivasyon 
fonksiyonları birer yazı biçiminde isimsel olarak girilebilir. Ya da tensorflow.keras.activations 
modülündeki fonksiyonlar olarak girilebilir. Örneğin:

from tensorflow.keras.layers import Dense
layer = Dense(100, activation='relu')

ya da örneğin:
    
from tensorflow.keras.activations import relu
layer = Dense(100, activation=relu)


Dense fonksiyonun use_bias parametresi default durumda True biçimdedir. Bu parametre 
katmandaki nöronlarda "bias" değerinin kullanılıp kullanılmayacağını belirtmektedir. 

Metodun kernel_initializer parametresi katmandaki nöronlarda kullanılan w parametrelerinin 
ilkdeğerlerinin rastgele biçimde hangi algoritmayla oluşturulacağını belirtmektedir. 
Bu parametrenin default değeri "glorot_unfiorm" biçimindedir. 

Metodun bias_initializer parametresi ise katmandaki nöronların "bias" değerlerinin 
başlangıçta nasıl alınacağını belirtmektedir. Bu parametrenin default değeri de 
"zero" biçimdedir. Yani bias değerleri başlangıçta 0 durumundadır.


Keras'ta Sequential modelde girdi katmanı programcı tarafından yaratılmaz. İlk 
saklı katman yaratılırken girdi katmanındaki nöron sayısı input_dim parametresiyle 
ya da input_shape parametresiyle belirtilmektedir. input_dim tek boyutlu girdiler için 
input_shape ise çok boyutlu girdiler için kullanılmaktadır. Örneğin:

layer = Dense(100, activation='relu', input_dim=8) # tek boyutlu 8 tane nörondan oluşuyor demek

input_shape= (10,10) # girdi katmanı 2 boyutlu 10'a 10'luk matris demek

Tabii input_dim ya da input_shape parametrelerini yalnızca ilk saklı katmanda kullanabiliriz. 
Genel olarak ağın girdi katmanında dataset_x'tekü sütun sayısı kadar nöron olacağına 
göre ilk katmandaki input_dim parametresini aşağıdaki gibi de girebiliriz:

layer = Dense(100, activation='relu', input_dim= training_dataset_x.shape[1])

Aslında Keras'ta girdi katmanı için tensorflow.keras.layers modülünde Input isminde 
bir katman da kullanılmaktadır. Tenseoflow'un yeni versiyonlarında girdi katmanının 
Input katmanı ile oluşturulması istenmektedir. Aksi takdirde bu yeni versiyonlar uyarı 
vermektedir. Girdi katmanını Input isimli katman sınıfıyla oluştururken bu Input 
sınıfının __init__ metodunun birinci parametresi bir demet biçiminde (yani shape olarak) 
girilmelidir. Örneğin:

input = Input((8, ))

Burada 8 nöronluk bir girdi katmanı oluşturulmuştur. Yukarıda da belirttiğimiz gibi 
eskiden ilk saklı katmanda girdi katmanı belirtiliyordu. Ancak Tensorflow kütüphanesinin 
yeni verisyonlarında ilk saklı katmanda girdi katmanının belirtilmesi artık uyarıya 
(warning) yol açmaktadır.

Her katmana istersek name parametresi ile bir isim de verebiliriz. Bu isimler model 
özeti alınırken ya da katmanlara erişilirken kullanılabilmektedir. Örneğin:

layer = Dense(100, activation='relu',  name='Hidden-1')

Oluşturulan katman nesnesinin model nesnesine eklenmesi için Sequential sınıfının 
add metodu kullanılmaktadır. Örneğin:

input = Input((8, ))
model.add(input)
layer = Dense(100, activation='relu',  name='Hidden-1')
model.add(layer)

Programcılar genellikle katman nesnesinin yaratılması ve eklenmesini tek satırda 
aşağıdaki gibi yaparlar:

model.add(Dense(100, activation='relu', input_dim=9, name='Hidden-1'))

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
3) Modele katmanlar eklendikten sonra bir özet bilgi yazdırılabilir. Bu işlem 
Sequential sınıfının summary isimli metoduyla yapılmaktadır. 

Yukarıda da belirttiğimiz gibi bir katmandaki "eğitilebilir (trainable)" parametrelerin 
sayısı aşağıda olan örnekteki gibi hesaplanmaktadır. Aşağıdaki modeli inceleyiniz:
  
    
model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

B modelde bir girdi katmanı, iki saklı katman (biz bunlara ara katman da diyeceğiz) 
bir de çıktı katmanı vardır. summary metodundan elde edilen çıktı şöyledir.

    Model: "Diabetes"
    ┌─────────────────────────────────┬────────────────────────┬───────────────┐
    │ Layer (type)                    │ Output Shape           │       Param # │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ Hidden-1 (Dense)                │ (None, 16)             │           144 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ Hidden-2 (Dense)                │ (None, 16)             │           272 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ Output (Dense)                  │ (None, 1)              │            17 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
    Total params: 433 (1.69 KB)
    Trainable params: 433 (1.69 KB)
    Non-trainable params: 0 (0.00 B)
    Trainable params: 433 (1.69 KB)
    Non-trainable params: 0 (0.00 B)

Burada ağımızdaki girdi katmanında 8 nöron olduğuna göre ve ilk saklı katmanda da 
16 nöron olduğuna göre ilk saklı katmana (8 * 16) nöron girmektedir. Öte yandan 
her nöronun bir tane bias değeri de olduğuna göre ilk katmandaki tahmin ayarlanması 
gereken parametrelerin (trainable parameters) sayısı (8 * 16 + 16 = 144) tanedir. 
İkinci saklı katmana 16 nöron dense biçimde bağlanmıştır. O halde ikinci saklı 
katmandaki ayarlanması gereken parametreler toplamda (16 * 16 + 16 = 272) tanedir. 
Modelimizin çıktı katmanında 1 nöron vardır. Önceki katmanın 16 çıkışı olduğuna 
göre bu çıktı katmanında (16 * 1 + 1 = 17) tane ayarlanması gereken parametre vardır. 

Ağın saklı katmanlarında en çok kullanılan aktivasyon fonksiyonu "relu" isimli fonksiyondur. 
İkili sınıflandırma problemlerinde çıktı katmanı tek nörondan oluşur ve bu katmandaki 
aktivasyon fonksiyonu "sigmoid" fonksiyonu olur. Sigmoid fonksiyonu 0 ile 1 arasında 
bir değer vermektedir. Biz aktivasyon fonksiyonlarını izleyen paragraflarda ele alacağız.

Aşağıdaki örnekte "dibates" veri kümesi üzerinde ikili sınıflandırma problemi için 
bir sinir ağı oluşturulmuştur. 

---------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
4) Model oluşturulduktan sonra modelin derlenmesi (compile edilmesi) gerekir. Buradaki 
"derleme" makine diline dönüştürme anlamında bir terim değildir. Eğitim için bazı 
belirlemelerin yapılması anlamına gelmektedir. Bu işlem Sequential sınıfının 
compile isimli metoduyla yapılmaktadır. Modelin compile metoduyla derlenmesi sırasında 
en önemli iki parametre "loss fonksiyonu" ve "optimizasyon algoritması"dır.

Eğitim sırasında ağın ürettiği değerlerin gerçek değerlere yaklaştırılması için w 
ve bias değerlerinin nasıl güncelleneceğine ilişkin algoritmalara "optimizasyon 
algoritmaları" denilmektedir. Matematiksel optimizasyon işlemlerinde belli bir 
fonksiyonun minimize edilmesi istenir. İşte minimize edilecek bu fonksiyona da 
"loss fonksiyonu" denilmektedir. Başka bir deyişle optimizasyon algoritması loss 
fonksiyonun değerini minimize edecek biçimde işlem yapan algoritmadır. Yani optimizasyon 
algoritması loss fonksiyonunu minimize etmek için yapılan işlemleri temsil etmektedir. 
    

Loss fonksiyonları ağın ürettiği değerlerle gerçek değerler arasındaki farklılığı 
temsil eden fonksiyonlardır. Loss fonksiyonları genel olarak iki girdi alıp bir 
çıktı vermektedir. Loss fonksiyonunun girdileri gerçek değerler ile ağın ürettiği 
değerlerdir. Çıktı değeri ise aradaki farklığı belirten bir değerdir. Eğitim sırasında 
gitgide loss fonksiyonun değerinin düşmesini bekleriz. Tabii loss değerinin düşmesi 
aslında ağın gerçek değerlere daha yakın değerler üretmesi anlamına gelmektedir.

Loss fonksiyonları çıktının biçimine yani problemin türüne bağlı olarak seçilmektedir. 
Örneğin ikili sınıflandırma problemleri için "binary cross-entropy", çoklu sınıflandırma 
problemleri için "categorical cross-entropy", lojistik olmayan regresyon problemleri 
için "mean squared error" isimli loss fonksiyonları tercih edilmektedir. 

Optimizasyon algoritmaları aslında genel yapı olarak birbirlerine benzemektedir. 
Pek çok problemde bu algoritmaların çoğu benzer performans göstermektedir. En çok 
kullanılan optimizasyon algoritmaları "rmsprop", "adam" ve "sgd" algoritmalarıdır. 
Bu algoritmalar "gradient descent" denilen genel optimizasyon yöntemini farklı 
biçimlerde uygulamaktadır.

compile metodunda optimizasyon algoritması bir yazı olarak ya da tensorflow.keras.optimizers 
modülündeki sınıflar türünden bir sınıf nesnesi olarak girilebilmektedir. Örneğin:

model.compile(optimizer='rmsprop', ...)

Örneğin:

from tensorflow.keras.optimizers import RMSprop

rmsprop = RMSprop()

model.compile(optimizer=rmsprop, ...)
---------------------------------------------------------------------------------
"""