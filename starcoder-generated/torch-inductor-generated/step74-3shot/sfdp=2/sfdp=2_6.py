
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        s1 = torch.matmul(query, key.transpose(-2, -1))
        s2 = s1.div(10000)
        s3 = torch.nn.functional.softmax(s2,dim=-1)
        s4 = torch.nn.functional.dropout(s3, p=0.1)
        o = torch.matmul(s4, value)
        return o

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
value = torch.randn(1, 64, 56, 56)
