
class Model(torch.nn.Module):
    def forward(self, I):
        p = torch.mm(I[:, 0:12], torch.t(I[:, 1:4]))
        I = torch.mm(torch.t(I[:12, :12]), I[12:, :])
        return p + I
# Inputs to the model
I = torch.randn(2100, 100)
