
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key_conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.value_conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)

    def forward(self, x1):
        x2 = self.key_conv(x1)
        x3 = self.value_conv(x1)
        v4 = torch.matmul(x2, x3.transpose(-2, -1))
        v5 = v4.div(0.12)
        v6 = v5.softmax(dim=-1)
        v7 = torch.nn.functional.dropout(v6, p=0.1, training=False) # dropout with p=0.1 when training=False
        v8 = torch.matmul(v7, x3)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
