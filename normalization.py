import numpy as np
import pandas as pd 
def z_scaling(array,columns,numerical_variables):
    var_list = []
    for i in numerical_variables:
        var_list.append(columns.index(i))
    for j in var_list:
        array[:,j] = (array[:,j]-array[:,j].mean())/(array[:,j].std())

