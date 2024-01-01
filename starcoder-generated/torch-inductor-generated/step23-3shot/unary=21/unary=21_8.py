
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_seq = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.ReLU())
    def forward(self, x):
        x = self.linear_relu_seq(x)
        v2 = torch.tanh(x)
        return v2
# Inputs to the model
x = torch.randn(10, 4)
