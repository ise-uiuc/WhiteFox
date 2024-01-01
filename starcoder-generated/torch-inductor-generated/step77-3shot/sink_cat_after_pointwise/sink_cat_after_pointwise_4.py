
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.transpose(1, -1).reshape(-1, 2).unsqueeze(0)  # x has shape (1, 64, 2)
        for i in range(5):
            x = torch.cat((x.transpose(1, -1).unsqueeze(-1), x), dim=-1)
        x = x.squeeze().reshape(-1, 2, 64).transpose(1, -1)  # x has shape (256, 2, 64)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
