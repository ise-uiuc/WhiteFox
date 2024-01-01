
class Model(torch.nn.Module):
    def __init__(self, input_channels=128):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(1)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, 0.5)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
input_channels = 128
m = Model(input_channels)

# Inputs to the model
x1 = torch.randn(1, 64, input_channels)
x2 = torch.randn(1, input_channels, 64)
