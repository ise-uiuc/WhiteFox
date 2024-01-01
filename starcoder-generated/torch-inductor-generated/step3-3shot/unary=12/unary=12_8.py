
class Model(torch.nn.Module):
    # Your code starts here. Make changes to the below __init__() and forward() methods. Your changes should meet all specifications mentioned in the "Description of requirements" section of this issue.
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v1)
        v4 = torch.sigmoid(v3)
        # Your code ends here.
        return v3 * v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
