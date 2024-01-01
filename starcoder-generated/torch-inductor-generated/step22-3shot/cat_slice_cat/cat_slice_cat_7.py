
import numpy as np
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *xs):
        xs_cat = torch.cat(xs, dim=1)
        l1 = list(range(np.iinfo(np.int64).max))
        sl1 = slice(0, 9223372036854775807)
        sl2 = slice(sl1.start, sl1.stop - 2)
        xs_cat_sliced = xs_cat[:, sl2]
        l1.extend(list(range(abs(sl1.stop) - sl1.start - 2)))
        xs_cat_sliced = torch.cat([xs_cat, xs_cat_sliced], dim=1)
        return xs_cat_sliced

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
