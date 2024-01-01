
class model(torch.nn.Module):
    def forward(self, x1):
        x1.clamp(1, 55)
        x1.contiguous()
        x2 = torch.transpose(x1, 0, 1).contiguous()
        x3 = x1.unsqueeze(0).contiguous()
        return x3
# Inputs to the model
x1 = torch.randn(2, 4)
