from copy import deepcopy
import sunpy.map


def _default_apply_function(data):
    return data


def _both_maps_same_dimension(m1, m2):
    return m1.dimensions == m2.dimensions


def _data_add(data1, data2):
    return data1 + data2


def _data_subtract(data1, data2):
    return data1 - data2


def _data_multiply(data1, data2):
    return data1 * data2


def _data_divide(data1, data2):
    return data1 / data2


def map_function_two_arguments(m1, m2, func=_data_add, use_meta=True):
    if not _both_maps_same_dimension(m1, m2):
        raise ValueError("Both input maps must have the same dimensions.")

    if use_meta:
        new_meta = deepcopy(m1.header)
    else:
        new_meta = deepcopy(m2.header)

    return sunpy.map.Map(func(deepcopy(m1.data), deepcopy(m2.data)), new_meta)


#
# Convenience functions
#
def map_add(m1, m2, use_meta=True):
    return map_function_two_arguments(m1, m2, func=_data_add, use_meta=use_meta)


def map_subtract(m1, m2, use_meta=True):
    return map_function_two_arguments(m1, m2, func=_data_subtract,
                                      use_meta=use_meta)
