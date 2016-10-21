import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
from multiprocessing import Pool


tv_train = pd.read_csv('data/timevarying_param_train.csv',header=None)
tv_train.columns = ['product_no','key_index','param_name','param_value','add_time']
tv_train = tv_train[tv_train.key_index>=0.85]
products = list(tv_train.product_no.unique())

def foo(product):
    this_product = tv_train[tv_train.product_no==product]
    this_product.add_time = this_product.add_time.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    add_time = int(this_product.add_time.min())
    key_index = float(this_product.iloc[0,:].key_index)
    return add_time,key_index

time_key = Pool(12).map(foo,products)
time = [i[0] for i in time_key]
key = [i[1] for i in time_key]
plt.scatter(time,key)
plt.show()


