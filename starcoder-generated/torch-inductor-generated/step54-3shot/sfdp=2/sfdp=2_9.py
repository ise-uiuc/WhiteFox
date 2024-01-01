
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(in_features=200,
                                                      out_features=200,
                                                      num_heads=2,
                                                      bias=False,
                                                      dropout=0.1)
 
    def forward(self, query, key, value):
        scaled_qk = self.attention(query, key, value)[0]
        return scaled_qk

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 200)
key = torch.randn(1, 2, 200)
value = torch.randn(1, 2, 200)
