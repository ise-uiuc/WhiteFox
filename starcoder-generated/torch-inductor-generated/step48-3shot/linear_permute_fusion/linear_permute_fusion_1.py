
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, input_tensor):
        t1 = torch.nn.functional.linear(input_tensor, self.linear.weight, self.linear.bias)
        t1_perm = t1.permute(0, 2, 1, 3)
        return t1_perm
# Inputs to the model
input_tensor = torch.randn(1, 2, 2, 2)
