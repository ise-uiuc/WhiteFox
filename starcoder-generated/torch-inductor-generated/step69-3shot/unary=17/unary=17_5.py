
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.nn.functional.dropout(v2)
        v4 = torch.nn.functional.elu(v3)
        v5 = torch.nn.functional.selu(v4)
        v6 = torch.tanh(v5)
        v7 = torch.nn.functional.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
