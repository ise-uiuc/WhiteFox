
# For simplicity, assume that we have two tensors to be added together
x2 = torch.randn(1, 1, 1, 256)
x3 = torch.randn(1, 1, 1, 256)

class Model(torch.nn.Module):
    def __init__(self, x2, x3):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
 
    def forward(self, input_tensor, x2=x2, x3=x3):
        v1 = self.fc1(input_tensor)
        v2 = x2 + x3
        v3 = v1 + v2
        return v3

m = Model(x2, x3)

