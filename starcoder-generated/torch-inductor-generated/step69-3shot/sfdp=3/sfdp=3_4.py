
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, query, key, value, scale_factor, dropout_p):
        k = query.mul(query)
        