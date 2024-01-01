
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        s1 = torch.matmul(q, k.transpose(-2, -1))
        s2 = s1 / inv_scale_factor
        f1 = torch.nn.functional.softmax(s2, dim=-1)
        output = torch.nn.functional.dropout(f1, p=dropout_p)
        output = torch.matmul(output, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 1, 128)
k = torch.randn(1, 8, 128, 64)
v = torch.randn(1, 8, 128, 64)
