
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(1280, 8)
        self.d = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        v = self.a(x)
        v = torch.mul(v, 1.0)
        v = torch.nn.functional.sigmoid(v)
        v = self.d(v)
        v = torch.nn.functional.relu(v)
        v = torch.nn.functional.leaky_relu(v, negative_slope=0.01)
        v = torch.nn.functional.softmax(v, dim=-1)
        return v
# Inputs to the model
x = torch.randn(1, 8, 1, 8)
