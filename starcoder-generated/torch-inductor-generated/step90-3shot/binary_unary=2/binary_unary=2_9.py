
class MyModule(torch.nn.Module):
    def forward(self, input : Tensor) -> Tensor:
        t1 = input / 2
        t2 = t1 + 3.5
        t13 = t2 - 1
        t3 = torch.relu(t13)
        return torch.squeeze(t3, -1)
# Inputs to the model
input = torch.randn(1, 64, 64)
