
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.key_conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.value_conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.scale_factor = 16
        self.dropout_p = 0.3
 
    def forward(self, x, y):
        v1 = self.query_conv(x)
        v2 = self.key_conv(y)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3 * self.scale_factor
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p = self.dropout_p)
        output = torch.matmul(v6, v3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16, 32, 32)
y = torch.randn(1, 16, 16, 16)
