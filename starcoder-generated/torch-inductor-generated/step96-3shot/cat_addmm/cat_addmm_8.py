
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = torch.Size([2, 3, 4])
    def forward(self, x):
        x = x.view(2, 3, 4)
        return x
# Inputs to the model
x = torch.ones(4, 3)
