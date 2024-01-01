
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), dim=1) # Comment out and change model to this line to observe a failure
        v1 = v1.unsqueeze(0)
        v2 = torch.cat((x1, x2), dim=1)
        y = v2.view(v1.shape[0], -1)
        return y
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 4)
