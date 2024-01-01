
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Unfold(kernel_size=(3, 9), stride=(2, 8), padding=(0, 7), dilation=(2, 1))
        self.m2 = torch.nn.Fold((4, 10), kernel_size=(3, 9), stride=(2, 8), padding=(0, 7), dilation=(2, 1))
    def forward(self, x):
        return self.m2(self.m1(x))
# Inputs to the model
x = torch.randn(20, 16, 6, 9)
