
class model(torch.nn.Module):
    def forward(self, input_r):
        t1 = torch.squeeze(input_r)
        t2 = torch.rand(t1.shape)
        return torch.sigmoid(t2)
# Inputs to the model
input_r = torch.randn(1, 2)
