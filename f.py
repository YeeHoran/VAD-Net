
import torch
# 假设 'tensor1' 和 'tensor2' 是两个张量
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

# 计算对应项差值的平方
squared_diff = (tensor1 - tensor2)**2

# 计算平方和
sum_squared_diff = squared_diff.sum()

# 开方得到欧氏距离
euclidean_distance = torch.sqrt(sum_squared_diff)

print("欧氏距离:", euclidean_distance.item())