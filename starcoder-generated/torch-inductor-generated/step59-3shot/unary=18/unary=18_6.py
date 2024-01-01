
class Model(torch.nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(size[0] * size[1] * size[2], dim)
        self.layer2 = torch.nn.Linear(dim, dim)
        self.layer3 = torch.nn.Linear(dim, 10)
    def forward(self, input):
        x = input.view(input.shape[0], -1)
        x = F.sigmoid(self.layer1(x))
        x = F.log_softmax(self.layer2(x), dim=-1)
        x = F.log_softmax(self.layer3(x), dim=-1)
        return x
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
