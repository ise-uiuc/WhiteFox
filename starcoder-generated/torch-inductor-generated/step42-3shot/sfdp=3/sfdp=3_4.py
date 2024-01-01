
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q2, k1, v4):
        s1 = q2.matmul(k1.transpose(-2, -1))
        s2 = s1 * scale_factor
        s3 = torch.nn.functional.softmax(s2, dim=-1)
        s4 = torch.nn.functional.dropout(s3, p=dropout_p)
        o1 = s4.matmul(v4)
        return o1
 
# Initializing the model
m = Model()

# Inputs to the model
q2 = torch.randn(1, 13, 64)
k1 = torch.randn(1, 37, 51)
v4 = torch.randn(1, 37, 64)
