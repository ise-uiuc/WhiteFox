
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(28 * 28, 1024)
        self.value = torch.nn.Linear(28 * 28, 1024)
 
    def forward(self, query):
        s1 = self.key(query)
        v1 = self.value(query)
        v2 = torch.matmul(v1, s1.transpose(-2, -1)) / 5000000
        v3 = v2.softmax(-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1)
        result = torch.matmul(v4, v1)
        return result

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 28 * 28)
