
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor):
        l = torch.nn.Linear(5, 5)
        output = input_tensor * torch.clamp_min(torch.nn.functional.linear(input_tensor, l.weight, l.bias), min_value=-0.5)
        return torch.clamp(torch.tanh(output), max_value=0.05)

# Inputs to the model
input0 = torch.randn(3, 5)
