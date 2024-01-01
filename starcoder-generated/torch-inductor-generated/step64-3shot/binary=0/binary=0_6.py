
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 20, (3, 5), stride=(2, 2), padding=(1, 2))
        self.conv2 = torch.nn.Conv2d(20, 40, 3, stride=1)
    def forward(self):
        return
# Inputs to the model
