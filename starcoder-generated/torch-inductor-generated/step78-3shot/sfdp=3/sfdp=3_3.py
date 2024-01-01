
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v):
        k1 = k * self.scale_factor
        w1 = torch.nn.functional.softmax(k1, dim=-1)
        dropout_w1 = torch.nn.functional.dropout(w1, p=self.dropout_p)
        output = dropout_w1.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 320, 32)
k = torch.randn(1, 320, 32)
v = torch.randn(1, 320, 32)
