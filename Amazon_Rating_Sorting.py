############################ İŞ PROBLEMİ #############################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış# sonrası verilen puanların
# doğru şekilde hesaplanmasıdır. Bu# problemin çözümü e-ticaret sitesi için daha fazla müşteri
# memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın # alanlar için sorunsuz bir
# alışveriş deneyimi demektir. Bir diğer # problem ise ürünlere verilen yorumların doğru bir
# şekilde # sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne # çıkması ürünün
# satışını doğrudan etkileyeceğinden dolayı hem # maddi kayıp hem de müşteri kaybına neden olacaktır.
# Bu 2 temel # problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.


######################## Veri Seti Hikayesi ##########################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# 12 Değişken   4915 Gözlem     71.9 MB

# reviewerID          = Kullanıcı ID’si
# asin                = Ürün ID’si
# reviewerName        = Kullanıcı Adı
# helpful             = Faydalı değerlendirme derecesi
# reviewText          = Değerlendirme
# overall             = Ürün rating’i
# summary             = Değerlendirme özeti
# unixReviewTime      = Değerlendirme zamanı
# reviewTime          = Değerlendirme zamanı Raw
# day_diff            = Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes         = Değerlendirmenin faydalı bulunma sayısı
# total_vote          = Değerlendirmeye verilen oy sayısı


# Veriyi Hazırlama ve Analiz Etme

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" %x)
pd.set_option("display.width", 500)

df = pd.read_csv("DATASETS/amazon_review.csv")
df.head(10)
#       reviewerID        asin                                      reviewerName helpful                                         reviewText  overall                                            summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
# 0  A3SBTW3WS4IQSN  B007WTAJTO                                               NaN  [0, 0]                                         No issues.  4.00000                                         Four Stars      1406073600  2014-07-23       138            0           0
# 1  A18K1ODH1I2MVB  B007WTAJTO                                              0mie  [0, 0]  Purchased this for my device, it worked as adv...  5.00000                                      MOAR SPACE!!!      1382659200  2013-10-25       409            0           0
# 2  A2FII3I2MBMUIA  B007WTAJTO                                               1K3  [0, 0]  it works as expected. I should have sprung for...  4.00000                          nothing to really say....      1356220800  2012-12-23       715            0           0
# 3   A3H99DFEG68SR  B007WTAJTO                                               1m2  [0, 0]  This think has worked out great.Had a diff. br...  5.00000             Great buy at this price!!!  *** UPDATE      1384992000  2013-11-21       382            0           0
# 4  A375ZM4U047O79  B007WTAJTO                                      2&amp;1/2Men  [0, 0]  Bought it with Retail Packaging, arrived legit...  5.00000                                   best deal around      1373673600  2013-07-13       513            0           0
# 5  A2IDCSC6NVONIZ  B007WTAJTO                                           2Cents!  [0, 0]  It's mini storage.  It doesn't do anything els...  5.00000                        Not a lot to really be said      1367193600  2013-04-29       588            0           0
# 6  A26YHXZD5UFPVQ  B007WTAJTO                                        2K1Toaster  [0, 0]  I have it in my phone and it never skips a bea...  5.00000                                         Works well      1382140800  2013-10-19       415            0           0
# 7  A3CW0ZLUO5X2B1  B007WTAJTO  35-year Technology Consumer "8-tracks to 802.11"  [0, 0]  It's hard to believe how affordable digital ha...  5.00000  32 GB for less than two sawbucks...what's not ...      1404950400  2014-10-07        62            0           0
# 8  A2CYJO155QP33S  B007WTAJTO                                         4evryoung  [1, 1]  Works in a HTC Rezound.  Was running short of ...  5.00000                                      Loads of room      1395619200  2014-03-24       259            1           1
# 9  A2S7XG3ZC4VGOQ  B007WTAJTO                                          53rdcard  [0, 0]  in my galaxy s4, super fast card, and am total...  5.00000                                        works great      1381449600  2013-11-10       393            0           0

df.info()
# RangeIndex: 4915 entries, 0 to 4914
# Data columns (total 12 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   reviewerID      4915 non-null   object
#  1   asin            4915 non-null   object
#  2   reviewerName    4914 non-null   object
#  3   helpful         4915 non-null   object
#  4   reviewText      4914 non-null   object
#  5   overall         4915 non-null   float64
#  6   summary         4915 non-null   object
#  7   unixReviewTime  4915 non-null   int64
#  8   reviewTime      4915 non-null   object
#  9   day_diff        4915 non-null   int64
#  10  helpful_yes     4915 non-null   int64
#  11  total_vote      4915 non-null   int64
# dtypes: float64(1), int64(4), object(7)

df.describe([0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])
#          overall   unixReviewTime   day_diff  helpful_yes  total_vote
# count 4915.00000       4915.00000 4915.00000   4915.00000  4915.00000
# mean     4.58759 1379465001.66836  437.36704      1.31109     1.52146
# std      0.99685   15818574.32275  209.43987     41.61916    44.12309
# min      1.00000 1339200000.00000    1.00000      0.00000     0.00000
# 10%      4.00000 1356825600.00000  167.00000      0.00000     0.00000
# 25%      5.00000 1365897600.00000  281.00000      0.00000     0.00000
# 50%      5.00000 1381276800.00000  431.00000      0.00000     0.00000
# 75%      5.00000 1392163200.00000  601.00000      0.00000     0.00000
# 80%      5.00000 1394582400.00000  638.00000      0.00000     0.00000
# 90%      5.00000 1400112000.00000  708.00000      0.00000     1.00000
# 95%      5.00000 1403308800.00000  748.00000      1.00000     1.00000
# 99%      5.00000 1404950400.00000  943.00000      3.00000     4.00000
# max      5.00000 1406073600.00000 1064.00000   1952.00000  2020.00000

df["overall"].value_counts()
# 5.00000    3922
# 4.00000     527
# 1.00000     244
# 3.00000     142
# 2.00000      80
# Name: overall, dtype: int64

df["overall"].hist()
# sola ya da sağa çarpık yapı wilson lower bound a uygun bir yapıdır.
# Varyasyon katsayısı:
# % 30 un altındaysa toplu veri
# % 30-50 arası dağınık veri
# % 50 üstü çok dağınık veri olarak adlandırılır.
# Varyasyon Katsayısı = 0.99685/4.58759=0.21 (std / mean)

# (Time_based_weighted_average)
# Tarihe göre ağırlıklı puan ortalamasını hesaplayalım.

# reviewTime değişkenini tarih değişkeni olarak tanıtarak reviewTime'ın max değerini current_date olarak kabul edelim.
# her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturalım.
# gün cinsinden ifade edilen  değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar)
# çeyrekliklerden gelen değerlere göre ağırlıklandırma yapalım.

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()
df["days"] = (current_date - df["reviewTime"]).dt.days

q1 = df["days"].quantile(0.25) # 280.0
q2 = df["days"].quantile(0.50) # 430.0
q3 = df["days"].quantile(0.75) # 600.0

df.loc[df["days"] <= q1, "overall"].mean() * 0.28 + \
    df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean() * 0.26 + \
    df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean() * 0.24 + \
    df.loc[df["days"] > q3, "overall"].mean() * 0.22
# Zaman Ağırlıklı Ort.  4.59559
# Genel Ort.            4.58759

#FArklı zaman dilimlerinin ortalamalarını karşılaştıralım.
a = df.loc[df["days"] <= q1, "overall"].mean()
b = df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean()
c = df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean()
d = df.loc[df["days"] > q3, "overall"].mean()

list = [a, b, c, d]

for index, l in enumerate(list, 1):
    print("Zaman Dilimi", index, ":", l)
# Zaman Dilimi 1 : 4.6957928802588995
# Zaman Dilimi 2 : 4.636140637775961
# Zaman Dilimi 3 : 4.571661237785016
# Zaman Dilimi 4 : 4.4462540716612375

# a > b > c > d Puan sıralaması
# Gün sayısı azaldıkça puan artmıştır!!!!!

# helpful_no değişkenini üretmemiz gerekiyor.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Eğer helpful_yes ve total vote değişkenleri verilmeseydi helpful değişkeni ile bunu bulabilirdik.
# 1. Yol
df["helpful"].head()
df["helpful"] = df["helpful"].str.strip("[ ]")
df["helpful_yes"] = df["helpful"].apply(lambda x: x.split(", ")[0]).astype(int)
df["total_vote"] = df["helpful"].apply(lambda x: x.split(", ")[1]).astype(int)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# 2. Yol
df["helpful_yes"] = df[["helpful"]].applymap(lambda x: x.split(", ")[0].strip("[")).astype(int)
df["total_vote"] = df[["helpful"]].applymap(lambda x: x.split(", ")[1].strip("]")).astype(int)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyelim.

# score_pos_neg_diff
def score_pos_neg_diff(pos, neg):
    return pos - neg

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
def wilson_lower_bound(pos, neg, confidence=0.95):
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2) # z tablo değeri = 0.975 = 1.96
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
# - z almamızın sebebi eğer +z alırsak yani üst tabanı alırsak verinin çok küçük bir kısmını görmüş olacaktık.
# Alt tabanı alarak verinin tamamını hesaba katmış oluyoruz.
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()

#  score_pos_neg_diff, score_average_rating ve wilson_lower_bound a göre ilk 20 yorumu belirleyelim.

df.sort_values("score_pos_neg_diff", ascending=False).head(20)
df.sort_values("score_average_rating", ascending=False).head(20)
df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Sonuçları yorumlayalım.

# Tüm yöntemleri incelediğimizde score_pos_neg_diff ve score_average_rating yönteminin verilen puana
# bağımlı bir yapıda olduklarını görüyoruz. wilson_lower_bound ise puandan ziyade kullanıcının ürüne
# yaptığı yorumun ne kadar faydalı bulunup, bulunmamısına bağlı bir sıralamadır. Yani burada puanı en
# düşük de olsa kullanıcı da yorumu faydalı ise wilson_lower_bound sıralamasına göre en yukarıda olabilir.