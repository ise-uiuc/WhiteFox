
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 16, 3, stride=1)
        self.maxpool = torch.nn.MaxPool2d(4)
        self.dropout = torch.nn.Dropout2d(p=0.2)
        self.relu = torch.nn.ReLU6()
        self.layernorm = torch.nn.LayerNorm([32, 16])
        self.gelu = torch.nn.GELU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.maxpool(v1)
        v3 = self.dropout(v2)
        v4 = self.relu(v3)
        v5 = self.layernorm(v4)
        v6 = self.gelu(v5)
        return torch.tanh(v6)
# Inputs to the model
x = torch.randn(1, 4, 48, 48)
