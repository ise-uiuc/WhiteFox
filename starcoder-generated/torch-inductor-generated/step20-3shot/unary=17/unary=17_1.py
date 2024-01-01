
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 1, stride=14)
        self.max_pool = torch.nn.MaxPool2d(3, 2, padding=0)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        y = self.conv_transpose(torch.relu(v1))
        v3 = self.max_pool(v1)
        w = self.conv_transpose1(v3)
        y1 = torch.relu(y)
        y2 = torch.relu(w)
        x = torch.max(y1, y2)
        return x
# Inputs to the model
x1 = torch.ones(1, 1, 1, 1)
