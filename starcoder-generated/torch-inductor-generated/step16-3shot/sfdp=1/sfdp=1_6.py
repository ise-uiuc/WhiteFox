
class Model(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.fc_k = torch.nn.Linear(d_model, d_k)
        self.fc_v = torch.nn.Linear(d_model, d_v)
        self.scaled_dot_product = ScaledDotProductAttention(temperature=d_k ** 0.5)
 
    def forward(self, x1, x2, x3):
        k = self.fc_k(x1)
        v = self.fc_v(x2)
        return self.scaled_dot_product(k, v, x3)

# Initializing the model
m = Model(100, 25, 40)

# Inputs to the model
x1 = torch.randn(3, 100) # query
x2 = torch.randn(2, 100) # key
x3 = torch.randn(1, 25) # value
