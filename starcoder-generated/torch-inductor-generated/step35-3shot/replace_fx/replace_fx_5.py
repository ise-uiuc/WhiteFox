
class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_0: torch.Tensor) -> torch.Tensor:
        tensor_1: torch.Tensor = torch.nn.functional.dropout(tensor_0, training=False)
        tensor_2: torch.Tensor = torch.nn.functional.dropout(tensor_0, training=True)
        output: torch.Tensor = tensor_1 + tensor_2
        return output
# Inputs to the model
x1 = torch.randn(10, 10)
