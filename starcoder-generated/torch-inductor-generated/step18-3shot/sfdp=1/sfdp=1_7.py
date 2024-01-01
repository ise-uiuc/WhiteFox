
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(3072, 10, bias=True)
 
    def forward(self, input, weight):
        v1 = torch.matmul(input, weight.transpose(-2, -1))
        v2 = v1 / 0.1
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.2)
        return self.model(v4)

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(256, 3072)
weight = torch.randn(2016, 3072)
