
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(input_tensor):
        A = input_tensor[:, :, :, :, :, :]
        B = input_tensor[:, :, :, :, :, :]
        C = input_tensor[:, :, :, :, :, :]
        D = input_tensor[:, :, :, :, :, :]
        return (A + B + C + D) / 4
# Inputs to the model
input_tensor = torch.randn(1, 64, 56, 56, 56, 56)
