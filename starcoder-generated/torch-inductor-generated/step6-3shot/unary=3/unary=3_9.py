
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = PIL.Image.open('dog.png')
x1 = torch.from_numpy(np.array(x1)).unsqueeze(0).float()
x1 = x1.permute([0, 3, 1, 2])
