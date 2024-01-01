
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.mean(x1, dim=(1, 2, 3))
        t1 = torch.nn.functional.softmax(v1, dim=-1)
        t2 = torch.unsqueeze(t1, dim=1)
        t3 = torch.unsqueeze(t1, dim=2)
        t4 = torch.unsqueeze(t1, dim=3)
        t5 = t2 * t3 * t4
        t6 = torch.transpose(t5, dim0=1, dim1=2)
        t7 = torch.transpose(t6, dim0=2, dim1=3)
        t8 = t7 - t5
        z9 = t1 - 0.5
        f10 = t8 + z9
        return f10
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
