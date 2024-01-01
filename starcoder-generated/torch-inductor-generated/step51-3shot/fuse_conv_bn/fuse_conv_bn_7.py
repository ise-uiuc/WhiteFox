
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(3, 10, 3)
        self.t2 = torch.nn.Dropout(0.2)
        self.t3 = torch.nn.ConvTranspose2d(10, 3, 3)
    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        # Dropout function contains random numbers. This randomness should be different
        # between executions, so we need to set a seed each time.
        torch.manual_seed(1)
        x = self.t3(x)
        torch.manual_seed(1)
        x = self.t3(x)
        return x
x = torch.randn(1, 3, 10, 10)
