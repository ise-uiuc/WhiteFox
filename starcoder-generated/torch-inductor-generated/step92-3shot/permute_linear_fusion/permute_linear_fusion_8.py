
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = (v2.unsqueeze(2).unsqueeze(3)).permute(0, 2, 1, 3, 4).squeeze(4).permute(0, 2, 3, 1, 4).squeeze(3)
        v4 = self.maxpool2d(v3)
        v5 = v4.squeeze()
        return v5 + v2
# Inputs to the model
input = torch.randn(1, 1, 2, 2)
