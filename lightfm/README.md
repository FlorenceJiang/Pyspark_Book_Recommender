**Explanation on lightfm file:**

This python script is for constructing a recommendation system via "lightfm" library: 
https://making.lyst.com/lightfm/docs/home.html 

However, due to **limited RAM of our computer and the usage of pd.pivot_table**, we are only able to run data of size 5000, which is far too small for comparison with our Spark ALS implementation. This makes the comparison between single-machine implementation and Spark ALS implementation ridiculous. Therefore, we gave up on this implementation and started working on Annoy Fast Search.

However, this lightfm implementation via python3 is able to process .csv files if provided sufficient RAM resources.
