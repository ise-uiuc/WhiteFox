
class Model(torch.nn.Module):
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, p=0.3)
        t2 = torch.nn.functional.softmax(t1, dim=-1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 64, 768)
