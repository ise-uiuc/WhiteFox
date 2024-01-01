
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value=None, scale_factor=1.0, dropout_p=0.2):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * 1
        v3 = v1.softmax(dim =_-1)
        v4 = F.dropout(v3, p =0.2)
        v5 = torch.matmul(v4, value)
        return v5
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 10, 10)
key = torch.randn(1, 3, 10, 10)
value = torch.randn(1, 3, 10, 10)
