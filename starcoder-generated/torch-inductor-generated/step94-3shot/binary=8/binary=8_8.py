
class Model(torch.nn.Module):
    def forward(self, input_tensor):
        return torch.nn.functional.relu(torch.nn.functional.relu(torch.nn.functional.relu(input_tensor)))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
