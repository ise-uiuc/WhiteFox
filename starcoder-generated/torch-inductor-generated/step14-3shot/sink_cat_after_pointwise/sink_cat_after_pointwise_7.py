
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(*[torch.nn.Conv2d(3, 10, kernel_size=5), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2,return_indices=not bool()), torch.nn.Conv2d(10, 20, kernel_size=5), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2,return_indices=not bool()), torch.nn.ReLU()]).eval()
    def forward(self, x):
        t1 = torch.cat([x, x], dim=1)
        t2 = t1.tanh()
        t3 = self.features(t2)
        y = t3
        return y
# Inputs to the model
x = torch.randn(2, 3, 10, 10)
