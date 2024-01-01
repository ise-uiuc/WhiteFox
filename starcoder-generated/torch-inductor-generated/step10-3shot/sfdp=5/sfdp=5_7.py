s
Model 1
class Model1(torch.nn.Module):
    def __init__(self, d_model=3, nhead=4, dropout_p=0.5):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout_p, bias=False)
 
    def forward(self, x):
        v1 = self.self_attn(x, x, x)
        return v1[0]

# Model 2
class Model2(torch.nn.Module):
    # This model inherits the forward() method from Model1
    pass

# Initializing the model
m = Model1(32, 4)

# Inputs to the model
x = torch.randn(10, 40, 32)
