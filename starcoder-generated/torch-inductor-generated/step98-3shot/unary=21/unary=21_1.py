
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, (4, 4))
        self.conv2 = torch.nn.Conv2d(20, 20, (1, 1))
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
    def forward(self, x) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.relu(y1)
        y3 = self.conv2(y2)
        y4 = self.tanh(y3)
        return y4   
# Inputs to the model
testInput = torch.randn(2, 10, 100, 100)
