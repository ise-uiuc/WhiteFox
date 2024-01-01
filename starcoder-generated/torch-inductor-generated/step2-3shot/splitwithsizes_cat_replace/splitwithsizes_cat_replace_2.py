
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
            torch.nn.Conv2d(8, 8, 3, stride=1, padding=1),
        ])
 
    def forward(self, x1):
        for conv in self.convs:
            x1 = conv(x1)
        split_tensor1, split_tensor2 = torch.split(x1, [32, 48], dim=1)
        x2 = torch.cat((split_tensor1, split_tensor2), dim=1)
        x3 = torch.max(x1, dim=1)[0]
        return True

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 128, 64, 64)
