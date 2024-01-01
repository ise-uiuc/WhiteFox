
if True:
    # pointwise convolution with kernel size 1
    t1 = x1
    # multiply by 0.5
    import math
    t2 = t1 * 0.5
if True:
    t1 = x1
    t2 = t1 * 0.5
    # pointwise convolution with kernel size 1
    import math
    t1 = t2 * 0.00390625
if True:
    t1 = x1
    t2 = t1 * 0.5
    t3 = t2 * 0.00390625
    # apply error function
    t4 = t3 * 0.00390625
    t5 = t4 + 1
if True:
    t1 = x1
    t2 = t1 * 0.5
    t3 = t2 * 0.00390625
    t4 = t3 * 0.00390625
    t5 = t4 + 1
    t6 = t5 * 0.5
    # multiply output of pointwise convolution and output of error function
if True:
    t1 = x1
    t2 = t1 * 0.5
    t3 = t2 * 0.00390625
    t4 = t3 * 0.00390625
    t5 = t4 + 1
    t6 = t5 * 0.5
    # multiply output of error function and constant '0.0022646326215007245'
if True:
    t1 = x1
    t2 = t1 * 0.5
    t3 = t2 * 0.00390625
    t4 = t3 * 0.00390625
    t5 = t4 + 1
    t6 = t5 * 0.5
    t7 = t6 * 0.0022646326215007245
