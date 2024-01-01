
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input[:, :, :], input[:, :, :])
        t2 = torch.mm(input[:, :, :], input[:, :, :])
        t3 = torch.mm(input[:, :, :], input[:, :, :])
        t4 = torch.mm(input[:, :, :], input[:, :, :])
        t5 = torch.mm(input[:, :, :], input[:, :, :])
        return t1 + t2 + t3 + t4 + t5
# Inputs to the model
input = torch.randn(20, 3, 4)
