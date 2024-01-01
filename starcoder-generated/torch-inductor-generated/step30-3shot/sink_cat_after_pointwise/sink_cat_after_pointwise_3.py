
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        y1 = torch.cat((x, x, x), dim=1).view(x.shape[0], -1)
        y2 = torch.relu(y1).view(x.shape[0], -1).tanh()
        y3 = (-y1).tanh()
        y4 = y2 + y3 + y1
        x = y4 * y4
        return x
# Inputs to the model
x = torch.tensor([1], dtype=torch.float32)
