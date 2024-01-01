
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 5), torch.nn.Sigmoid(), torch.nn.Conv2d(1, 1, 5), torch.nn.Dropout(p=.5), torch.nn.Softmax(), torch.nn.Flatten(1, -1), torch.nn.Sequential(torch.nn.Conv2d(2, 1, 5), torch.nn.Sigmoid(), torch.nn.Conv2d(1, 1, 5), torch.nn.Sigmoid()))
    def forward(self, x1):
        v1 = self.model1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 20, 20)
