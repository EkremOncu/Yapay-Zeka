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
a = np.random.randint(1, 10, (20, 10))
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

C rassal değişkeni "rastgele seçilen bir rengin RGB değerlerinin ortalamasını" 
belirtiyor olsun. Bu durumda her rengin bir RGB ortalaması vardır. Bu fonksiyon 
belli bir rengi alıp onun ortalamasını belirten bir sayıya eşlemektedir. 

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
çıkmaktadır. Bu nednele biz sürekli rassal değişkenler ve onların olasılıkları 
üzerinde biraz daha duracağız. 

Sürekli bir rassal değişkenin aralıksal olasılıklarını hesaplama aslında bir 
"intergral" hesabı akla getirmektedir. İşte sürekli rassal değişkenlrin aralıksal 
olasılıklarının hesaplanması için kullanılan fonksiyonlara "olasılık yoğunluk 
fonksiyonları (probability density functions)" denilmektedir. Birisi bize bir 
rassal değişkenin belli bir aralıktaki olasılığını soruyorsa o kişiin bize o 
rassal değişkene ilişkin "olasılık yoğunluk fonksiyonunu" vermiş olması gerekir. 
Biz de örneğin P{x0 < X < x1} olasılığını x0'dan x1'e f(x)'in integrali ile elde 
ederiz. 

Bir fonksiyonun olasılık yoğunluk fonksiyonu olabilmesi için -sonsuzdan + sonsuze 
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
kadar tüm birikimli olsılıkları veren fonksiyondur. Genellikle F harfi gösterilmektedir. 
Mrneğin F(x0) aslında P{X < x0} anlamına gelmektedir. Normal dağılımda F(x0) değeri 
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

Şöyle bir soru sorulduğunu düşünelim: "İnsanların zekaları ortalaması 100, standart 
sapması 15 olan normal dağılıma uygundur. Bu durumda zeka puanı 140'ın yukarısında 
olanların toplumdaki yüzdesi nedir?". Bu soruda istenen şey aslında normal dağılımdaki
P{X > 140} olasılığıdır. Yani x ekseninde belli bir noktanın sağındaki kümülatif 
alan sorulmaktadır. Bu alanı veren doğrudan bir fonksiyon olmadığı için bu işlem 
1 - F(140) biçiminde ele alınarak sonuç elde edilebilir. Yani örneğin:

nd = statistics.NormalDist(100, 15)
result = 1 - nd.cdf(140)

------------------------------------------------------------------------------------
"""


