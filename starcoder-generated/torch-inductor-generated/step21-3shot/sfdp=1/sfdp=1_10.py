
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Linear(in_features=768, out_features=768, bias=True)
        self.value = torch.nn.Linear(in_features=768, out_features=768, bias=False)
 
    def forward(self, x1, x2):
        k = self.qk(x1)
        v = self.value(x1)
        v1 = torch.matmul(x2, k.transpose(-2, -1))
        v2 = v1.div(0.0625)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1, training=True)
        o1 = v4.matmul(v)
        return o1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 768)
x2 = torch.randn(2, 1, 768)
