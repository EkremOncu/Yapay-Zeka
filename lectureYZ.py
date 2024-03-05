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
"""

















