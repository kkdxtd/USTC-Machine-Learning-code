
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt # 导入 matplotlib 用于绘图

POP_SIZE        = 60   # 种群大小
MIN_DEPTH       = 2    # 最小初始随机树深度
MAX_DEPTH       = 5    # 最大初始随机树深度
GENERATIONS     = 250  # 进化运行的最大代数
TOURNAMENT_SIZE = 5    # 锦标赛选择中锦标赛的大小
XO_RATE         = 0.8  # 交叉率 
PROB_MUTATION   = 0.2  # 每节点变异概率

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
FUNCTIONS = [add, sub, mul]
TERMINALS = ['x', -2, -1, 0, 1, 2] 

def target_func(x): # 进化的目标
    return x*x*x*x + x*x*x + x*x + x + 1

def generate_dataset(): # 从 target_func 生成 101 个数据点
    dataset = []
    for x in range(-100,101,2): 
        x /= 100
        dataset.append([x, target_func(x)])
    return dataset

class GPTree:
    def __init__(self, data = None, left = None, right = None):
        self.data  = data
        self.left  = left
        self.right = right
        
    def node_label(self): # 字符串标签
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # 文本输出
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x): 
        if (self.data in FUNCTIONS): 
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif self.data == 'x': return x
        else: return self.data
            
    def random_tree(self, grow, max_depth, depth = 0): # 使用 grow 或 full 方法创建随机树
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # 中间深度，grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)            
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION: # 在此节点变异
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation() 

    def size(self): # 树的大小（节点数）
        if self.data in TERMINALS: return 1
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self): # count 是列表以便“按引用”传递
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # 注意：count 是列表，因此是“按引用”传递
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # 返回以此为根的子树
                return self.build_subtree()
            else: # 在此处粘贴子树
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)  
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # 在随机节点交叉 2 棵树
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None) # 第二个随机子树
            self.scan_tree([randint(1, self.size())], second) # 第二个子树“粘贴”到第一棵树中
# end class GPTree
                   
def init_population(): # 分层半半初始化
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t) 
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t) 
    return pop

def fitness(individual, dataset): # 数据集上反向平均绝对误差归一化到 [0,1]
    return 1 / (1 + mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]))
                
def selection(population, fitnesses): # 使用锦标赛选择法选择一个个体
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # 选择锦标赛参赛者
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]]) 
            
def main():      
    # 初始化
    seed() # 初始化随机数生成器的内部状态
    dataset = generate_dataset()
    population= init_population() 
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

    # 开始进化！
    for gen in range(GENERATIONS):        
        nextgen_population=[]
        for i in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population=nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses),3), ", best_of_run:") 
            best_of_run.print_tree()

            # 可视化当前最佳公式
            x_values = [ds[0] for ds in dataset] # 使用数据集中的 x 值
            # 计算目标函数和当前最佳公式的 y 值
            target_y = [target_func(x) for x in x_values]
            best_of_run_y = [best_of_run.compute_tree(x) for x in x_values]

            plt.figure(figsize=(10, 6)) # 创建一个新的图表
            plt.plot(x_values, target_y, label='目标函数', color='blue', linestyle='--') # 绘制目标函数
            plt.plot(x_values, best_of_run_y, label='当前最佳公式', color='red') # 绘制当前最佳公式
            plt.title(f'第 {gen} 代 - 适应度: {round(best_of_run_f, 3)}') # 设置图表标题
            plt.xlabel('x 值') # 设置 x 轴标签
            plt.ylabel('y 值') # 设置 y 轴标签
            plt.legend() # 显示图例
            plt.grid(True) # 显示网格
            # 保存图表为图片文件
            plt.savefig(f'generation_{gen:03d}_fitness_{round(best_of_run_f, 3)}.png') 
            plt.close() # 关闭图表以释放内存
            print(f"已保存第 {gen} 代的公式可视化图到 generation_{gen:03d}_fitness_{round(best_of_run_f, 3)}.png")

        if best_of_run_f == 1: break   
    
    print("\n\n_________________________________________________\n运行结束\n最佳运行在第 " + str(best_of_run_gen) +\
          " 代达到，f=" + str(round(best_of_run_f,3)))
    best_of_run.print_tree()
    
if __name__== "__main__":
  main()
