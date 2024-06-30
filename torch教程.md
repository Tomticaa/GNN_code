* super关键字：用于继承父类，目的是为了提高代码维护性，在代码维护过程中若要求改父类名字，则不需大量修改其子类代码内容：

```
class Base:
    def __init__(self):
        print("Base initializer")

    def hello(self):
        print("Hello from Base")

class Derived(Base):
    def __init__(self):
        super().__init__()  # 调用父类的 __init__ 方法
        print("Derived initializer")

    def hello(self):
        super().hello()  # 调用父类的 hello 方法
        print("Hello from Derived")

d = Derived()  # 输出 Base initializer 和 Derived initializer
d.hello()  # 输出 Hello from Base 和 Hello from Derived

```

* Sequential类：作为网络模型的封装方法，用来按顺序封装多个网络层。使用 Sequential 容器可以方便地将多个层堆叠在一起，形成一个简洁的模块。

当你使用 `nn.Sequential` 来构建模型时，你不需要显式地定义 `forward` 方法。`nn.Sequential` 自动创建了一个模型，其中每个模块的输出顺序地作为下一个模块的输入，直至最后一个模块。这意味着 `forward` 方法已经被内置在 `nn.Sequential` 的实现中。为什么不需要 `forward` 方法：

* **自动连接**：`nn.Sequential` 容器负责处理所有层的前向传播。当你通过 `nn.Sequential` 添加模块时，它按照你添加它们的顺序自动管理数据的流动。
* **简化实现**：这种方式简化了网络的实现，使得定义线性的、无分支的前向传递网络变得非常直接和清晰。

这个函数只能实现模型层的顺序链接，即便方向传播简单，不需要定义forward函数，但是只能实现简单的层序连接，要实现复杂的跳跃连接时不可用。

```
import torch

N, D_in, H, D_out = 64, 1000, 100, 10

torch.manual_seed(1)
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# -----changed part-----#
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
print(model)
输出为：Sequential(
  (0): Linear(in_features=1000, out_features=100, bias=True)
  (1): ReLU()
  (2): Linear(in_features=100, out_features=10, bias=True)
)
```

* torch.manual_seed(1):使得接下来得到的随机矩阵元素围绕数字1生成；
* nn.Parameter()：将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。即在定义网络时这个tensor就是一个可以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优秀；
*
