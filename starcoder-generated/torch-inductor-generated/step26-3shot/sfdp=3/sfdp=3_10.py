
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        k = torch.tensor([
            [2.8343e-02, 1.0301e-02],
            [1.9309e-01, 4.9428e-02],
            [1.0161e-03, 3.6473e-03],
        ])
        scale = torch.tensor(10.0)
        v2 = torch.matmul(v1, k) * scale
        v3 = torch.nn.functional.softmax(v2, dim=0)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = torch.matmul(v4, k.transpose(0, 1))
        output = v5 * scale
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 2)
