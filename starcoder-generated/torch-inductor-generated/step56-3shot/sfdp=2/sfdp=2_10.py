
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(256, 128)
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = nn.functional.dropout(v0)
        q = v1
        k = v1
        v = v1
        result = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(128)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
