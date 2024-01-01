
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(32, 8, bias=False)
        self.key = torch.nn.Conv1d(768, 32, 1)
 
    def forward(self, x1, x2):
        v1 = self.query(x1)
        v2 = self.key(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.div(0.0625)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.2)
        output = v5.matmul(x2)
        return output

# Initializing the model
b = Model()

# Input to the model
x1 = torch.randn(2, 32)
x2 = torch.randn(16, 768, 128)
