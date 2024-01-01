
t1 = torch.nn.functional.linear(torch.randn(1, 2, 3), torch.randn(3, 2, 4), torch.randn(8))
t2 = t1.permute(0, 2, 1)

