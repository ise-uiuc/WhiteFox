
class Model(torch.nn.Module):
    def __init__(self, q, k, v, scale_factor, dropout_p):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v

    def dot_product_score(self, query, key):
        