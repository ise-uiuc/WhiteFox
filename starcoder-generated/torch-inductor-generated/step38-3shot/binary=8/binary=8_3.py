
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 20)
        self.linear2 = torch.nn.Linear(20, 8)
    def forward(self, input_tensor):
        return self.linear2(torch.tanh(self.linear1(input_tensor)))
# Inputs to the model
input_tensor = torch.randn(1, 5)
