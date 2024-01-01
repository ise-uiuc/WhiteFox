
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=2, dropout_p=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, heads * out_channels, 1, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        b, _, h, w = v1.shape
        v1 = v1.reshape(b, -1, h, w)
        c = v1.size(1)
        v2 = x2.transpose(1, 2)
        v3 = v1.mul(v2)
        v4 = torch.nn.functional.softmax(v3, dim=2)
        v5 = self.dropout(v4)
        v6 = v5.matmul(v1)
        return v6

# Initializing the model
m = Model(in_channels=3, out_channels=64, heads=2, dropout_p=0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 64, 64)
