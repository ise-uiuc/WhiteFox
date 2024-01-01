
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(3, 8)
        self.key = torch.nn.Linear(3, 8)
        self.value = torch.nn.Linear(3, 8)
        self.scale_factor = 2 ** 10.5
        self.dropout_p = 0.5
 
    def forward(self, x1):
        v1 = self.query(x1)
        v2 = self.key(x2)
        v3 = self.value(x1)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4 * self.scale_factor
        v6 = v5.softmax(dim=-1)
        v7 = torch.nn.functional.dropout(v6, self.dropout_p)
        v8 = v7.matmul(v8)
        return v8
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
