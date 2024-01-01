
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1024
        self.seq_len = 128
        self.dim = 64
    def forward(self, input1, input2, input3, input4):
        concat3 = torch.cat((input1, input4), 1)
        batch1 = torch.flatten(concat3, 1)
        matmul1 = torch.matmul(batch1, input3)
        output1 = torch.reshape(matmul1, (1, 128, 1024, 1))
        return output1
# Inputs to the model
input1 = torch.randn(1, 256, 1, 1)
input2 = torch.randn(1, 256, 1, 1)
input3 = torch.randn(256, 1)
input4 = torch.randn(1, 256, 1, 1)
