
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.unsqueeze(dim=1).unsqueeze(dim=1)
        y1 = y.tanh().sigmoid()
        y2 = y.tanh().relu()
        return torch.cat((y1, y2), dim=1).tanh() if y.shape!= (2, 3, 4, 4) else torch.cat((y1, y2), dim=1).relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
