import numpy as np
from matplotlib import pyplot as plt
import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops


img = basic_ops.load_image('images/obama.pgm')
img = filters.gray_scale_filter(img)

print(f'Median: {stat_ops.median(img)}')

print(f'Mittelwert: {stat_ops.mean(img)}')

print(f'Varianz: {stat_ops.variance(img)}')

print(f'Standard-Abweichung: {stat_ops.std(img)}')

print(f'Entropie: {stat_ops.entropy(img)}')