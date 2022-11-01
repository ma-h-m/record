def is_valid(point,m):
    if point[0] < 0:
        return 0
    if point[0] >= m.size[0]:
        return 0
    if point[1] < 0:
        return 0
    if point[1] >= m.size[1]:
        return 0
    return 1

# action = (('N',(-1,0)),('E',(0,1)),('S',(1,0)),('W',(0,-1)))
action = (('^',(-1,0)),('>',(0,1)),('V',(1,0)),('<',(0,-1)))

class map:
    def __init__(self):
        self.map = [[]]
        self.size = (0,0)
        self.st = (-1,-1)
        self.ed = (-1,-1)
    def create_map(self):
        self.size = (8,8)
        self.st = (2,0)
        self.ed = (6,7)
        self.map = [[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1],[0,0,1,1,0,1,0,1],[1,0,0,1,1,0,0,1],[1,1,0,0,1,0,1,1],[1,0,1,0,1,0,0,1],[1,0,0,0,0,1,0,0],[1,1,1,1,1,1,1,1]]
    def print_map(self):
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                print(self.map[i][j],end='')
            print()
    def print_map_and_value(self,value):
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if ((i,j) == m.ed):
                    print('G\t',end = '')
                    continue

                if self.map[i][j] == 1:
                    print('1\t',end = '')
                    continue
                tmp = -1
                rec = ()
                for k in action:
                    new_position = (i + k[1][0],j + k[1][1])
                    if (is_valid(new_position,m) == 1):
                        if (tmp < (m.ed == new_position) + alpha * values[new_position[0]][new_position[1]]):
                            tmp = (m.ed == new_position) + alpha * values[new_position[0]][new_position[1]]
                            rec = k[0]
                if (i,j) == m.st:
                    print('S' + rec + '\t',end = '')
                    continue
                print(rec + '\t',end='')
            print()
            
import numpy as np

# 初始化图
m = map()
m.create_map()

# 初始化价值地图，将地图上每一个点全部映射到一个float，作为它的价值估计
# 参数discount（alpha）设置为0.9，到达目标reward设置为1
# 将墙壁处的reward设置为-1，这样就一定不会走到墙壁中去，因为任何可以行走的部分的value必定非负

values = np.zeros(m.size)

alpha = 0.9


# 记录上一次迭代的价值函数
values_rec = np.ones(m.size)
import copy

while ((values != values_rec).any()):
    values_rec = copy.deepcopy(values)
    for i in range(m.size[0]):
        for j in range(m.size[1]):
            if (m.map[i][j] == 1):
                continue
            tmp = -1
            for k in action:
                new_position = (i + k[1][0],j + k[1][1])
                if (is_valid(new_position,m) == 1):
                    tmp = max(tmp,(m.ed == (i,j)) + alpha * values_rec[new_position[0]][new_position[1]])
            values[i][j] = tmp
    print('------------------------------------------------------------')
    m.print_map_and_value(values)



