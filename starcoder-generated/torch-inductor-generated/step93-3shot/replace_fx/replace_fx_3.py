
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 2, 1, 0)
        self.conv1.register_backward_hook(self.backward_hook1)
        self.conv2 = torch.nn.Conv2d(2, 2, 2, 1, 0)
        self.conv2.register_backward_hook(self.backward_hook2)
        self.conv3 = torch.nn.Conv2d(2, 2, 2, 1, 0)
    def backward_hook1(self, grad_output):
        print("Backward hook 1 called.")
    def backward_hook2(self, grad_output):
        print("Backward hook 2 called.")
    def forward(self, x1):
        return self.conv1(input=x1, weight=self.conv2(x1).detach())
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
