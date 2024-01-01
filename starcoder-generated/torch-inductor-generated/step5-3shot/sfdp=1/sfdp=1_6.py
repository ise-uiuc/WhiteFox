
class Model(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
 
    def forward(self, k, v, q, scale_factor, dropout_p):
        scale = 1 / scale_factor
        a = torch.matmul(q, k.transpose(-2, -1))
        a = a * scale
        b = F.softmax(a, dim=-1)
        c = F.dropout(b, p=dropout_p, training=self.training)
        d = torch.matmul(c, v)
        return d

# Initializing the model
d_model = 64
m = Model(d_model)

# Inputs to the model
k, v, q = (torch.randn(d_model, 1, 32), torch.randn(d_model, 1, 32), torch.randn(d_model, 1, 32))
dropout_p = 0.4
scale_factor = 0.001
