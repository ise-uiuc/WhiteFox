
class MHA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(128, 64)
        self.key = torch.nn.Linear(256, 128)
        self.value = torch.nn.Linear(256, 128)

    def forward(self, x2, x3, x4, x5, x6, x7, x8):
        v9 = self.query(x2).unsqueeze(-3)
        v10 = self.key(x3).transpose(-2, -1)
        v11 = torch.matmul(v9, v10)
        v12 = v11.div(x8)
        v13 = torch.nn.functional.softmax(v12, dim=1)
        v14 = torch.nn.functional.dropout(v13, p=x6, training=x5)
        v15 = self.value(x4)
        v16 = v14.matmul(v15)
        return v16

# Initializing the model
m = MHA()

# Inputs to the model
x2 = torch.randn(1, 128)
x3 = torch.randn(1, 96, 128)
x4 = torch.randn(1, 96, 128)
x5 = True
x6 = 0.3
x7 = False
x8 = 2
