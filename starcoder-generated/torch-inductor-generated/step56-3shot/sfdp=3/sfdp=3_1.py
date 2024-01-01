
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1/sqrt(512)
        self.dropout_p = 0
 
    def forward(self, query, key, value):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.mul(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = v4.matmul(value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 12, 256)
key = torch.randn(8, 12, 512)
value = torch.randn(8, 256, 512)
