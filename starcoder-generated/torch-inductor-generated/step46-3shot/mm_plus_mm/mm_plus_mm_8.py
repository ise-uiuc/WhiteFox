
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        batch_size = input1.shape[0]
        t1 = torch.mm(input1, input2)
        t2 = torch.zeros([batch_size, batch_size], dtype=torch.float)
        for x in range(batch_size):
            for y in range(batch_size):
                t2[x][y] = x + y
        t2 = torch.mm(t2, input3)
        t3 = torch.mm(input4, t2)
        return t1 - t3
# Inputs to the model
N = 10
input1 = torch.randn(N, N)
input2 = torch.randn(N, N)
input3 = torch.randn(N, N)
input4 = torch.randn(N, N)
