import os
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from umap import UMAP
import umap.plot
import matplotlib.pyplot as plt
import time
import datetime

# config spark session
exec(open(os.path.join(os.environ["SPARK_HOME"], 'python/pyspark/shell.py')).read())
app_name = 'umapVisualization'
master = 'local'
spark = SparkSession.builder.appName(app_name).master(master).getOrCreate()

# load data
genre_path = 'goodreads_book_genres_initial.json'
book_id_map_path = 'book_id_map.parquet'
itemFactors_path = 'itemFactors_10_0.01_100_percent.parquet'

genre = spark.read.json(genre_path)
book_id_map = spark.read.parquet(book_id_map_path)
itemFactors = spark.read.parquet(itemFactors_path)

def find_key_with_max_value(dic):
    if np.sum(np.array(list(dic.values())) != None) == 0:
        return None
    else:
        for k, v in dic.items():
            if v is None:
                dic[k] = 0
        return max(dic, key = dic.get)

genre.show(5)

# extract the genre with highest value for each book
genre_array = genre.collect()
bookid2genre = {genre_array[i][0]: find_key_with_max_value(genre_array[i][1].asDict()) for i in range(len(genre_array))}
bookid_genre_df = pd.DataFrame({'book_id': list(bookid2genre.keys()), 'genre': list(bookid2genre.values())})
bookid_genre_spark_df = spark.createDataFrame(bookid_genre_df)
id_genre_spark_df = bookid_genre_spark_df.join(book_id_map, on = 'book_id', how = 'inner').drop('book_id').withColumnRenamed('book_id_csv', 'id')

# id is string type, we change it to int for later join
id_genre_spark_df = id_genre_spark_df.withColumn('id', id_genre_spark_df['id'].cast(IntegerType()))

# explode item factors
itemFactors_exploded = itemFactors.select("id", itemFactors.features[0], itemFactors.features[1], itemFactors.features[2], itemFactors.features[3], itemFactors.features[4], \
                                                itemFactors.features[5], itemFactors.features[6], itemFactors.features[7], itemFactors.features[8], itemFactors.features[9])

id_itemFactor_spark_df = id_genre_spark_df.join(itemFactors_exploded, on = 'id', how = 'inner' )

# extract factors as 10-dimension vectors
data = np.array(id_itemFactor_spark_df.select('features[0]', 'features[1]', 'features[2]', 'features[3]', 'features[4]', 'features[5]', 'features[6]', 'features[7]', 'features[8]', 'features[9]').collect())

# extract labels, some book doesn't have any label so change that to 'none' genre
labels = np.array(['none' if label is None else label for label in [row[0] for row in id_itemFactor_spark_df.select('genre').collect()]])

# randomly draw 400k samples from data and labels for computation efficiency
np.random.seed = 66
samples_indices = np.random.choice(np.arange(data.shape[0]), size = 400000, replace = False)
sample_data, sample_labels = data[samples_indices], labels[samples_indices]

# visualize using UMAP
start_time = time.time()
mapper = UMAP(random_state=66, n_components=2, verbose = True).fit(sample_data)
umap.plot.points(mapper, labels = sample_labels, theme = 'fire')
plt.savefig('10_0.01_100_percent_400k_66.eps')
end_time = time.time()
print(str(datetime.timedelta(seconds = end_time - start_time)))


