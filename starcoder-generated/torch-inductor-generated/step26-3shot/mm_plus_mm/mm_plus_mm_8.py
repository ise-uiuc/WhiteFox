
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        # 1
        t1 = torch.matmul(input1, input2)
        t2 = torch.matmul(input3, input4)
        # 2
        t1 = torch.matmul(input1, input4)
        t2 = torch.matmul(input3, input2)
        # 3
        t1 = torch.matmul(input3, input4)
        t2 = torch.matmul(input1, input2)
        # 4
        t1 = torch.matmul(input1, input3)
        t2 = torch.matmul(input2, input4)
        t3 = t1 + t2
        # 5
        t1 = torch.matmul(input1, input3)
        t2 = torch.matmul(input2, input4)
        t3 = t1 * t2

        t1 = torch.matmul(input3, input1)
        t2 = torch.matmul(input2, input4)
        t4 = t1 + t2
        # 6
        t1 = torch.matmul(input3, input1)
        t2 = torch.matmul(input2, input4)
        t4 = t1 * t2

        t1 = torch.matmul(input3, input1)
        t2 = torch.matmul(input4, input2)
        t5 = t1 + t2
        # 7
        t1 = torch.matmul(input3, input1)
        t2 = torch.matmul(input4, input2)
        t5 = t1 * t2

        t1 = torch.matmul(t3, t4) + t5
        # 8
        t1 = torch.matmul(t3, t4) * t5

        t1 = torch.matmul(t4, t5) + t3
        t2 = torch.matmul(t5, t3) + t4
        t3 = torch.matmul(t3, t4) + t5
        t4 = torch.matmul(t4, t5) + t3
        t5 = torch.matmul(t3, t4) + t5
        t6 = torch.matmul(t4, t5) + t3

        return t1 + t2 + t3 + t4 + t5 + t6
# Inputs to the model
input1 = torch.randn(384, 768)
input2 = torch.randn(768, 384)
input3 = torch.randn(768, 384)
input4 = torch.randn(768, 768)
