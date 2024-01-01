
class Model(torch.nn.Module):
    def forward(self, inputs):
        m1 = torch.mm(inputs[0], inputs[0])
        m2 = torch.mm(inputs[3], inputs[0])
        return torch.mm(m1, m2)
# Input to the model
inputs = [
    torch.randn(7, 7),
    torch.randn(7, 7),
    torch.randn(7, 7),
    torch.randn(7, 7),
]
