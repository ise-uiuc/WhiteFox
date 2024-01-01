
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(8, 1)
        self.flatten = torch.flatten
        self.permute = torch.Tensor.transpose
    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x.transpose(0, 1), start_dim=0, end_dim=1)
        x = self.permute(x, 0, 1)
        return x
# Inputs to the model
x = torch.randn(8, 4)
