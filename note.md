# <center>粗略的笔记
## <center>代码框架

```shell
.
├── LICENSE
├── README.md
├── data
│   ├── cub
│   │   └── split
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── val.csv
│   └── miniimagenet
│       ├── download.sh
│       └── split
│           ├── test.csv
│           ├── train.csv
│           └── val.csv
├── model
│   ├── __init__.py
│   ├── data_parallel.py
│   ├── dataloader
│   │   ├── CUB
│   │   │   └── split
│   │   │       ├── test.csv
│   │   │       ├── train.csv
│   │   │       └── val.csv
│   │   ├── cub.py
│   │   ├── mini_imagenet.py
│   │   ├── samplers.py
│   │   ├── split_cub.py
│   │   ├── tiered_imagenet.py
│   │   └── transforms.py
│   ├── logger.py
│   ├── models
│   │   ├── INSTA.py
│   │   ├── INSTA_ProtoNet.py
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fcanet.py
│   │   ├── protonet.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── embedder.py
│   │       ├── stochastic_depth.py
│   │       ├── tokenizer.py
│   │       └── transformers.py
│   ├── networks
│   │   ├── __init__.py
│   │   ├── dropblock.py
│   │   ├── res10.py
│   │   ├── res12.py
│   │   ├── res18.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── embedder.py
│   │       ├── stochastic_depth.py
│   │       ├── tokenizer.py
│   │       └── transformers.py
│   ├── trainer
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fsl_trainer.py
│   │   └── helpers.py
│   └── utils.py
├── train_fsl.py
└── visual
    ├── concept.png
    ├── heatmap.png
    └── pipeline.png

15 directories, 51 files
```

## <center>文件功能
### <center>model
#### <center>logger.py

定义了一个日志记录系统，结合了JSON和TensorBoard进行数据记录。

#### <center>utils.py

在小样本学习算法的源代码中，`utils.py`这类文件通常是一个实用工具模块，包含了一系列辅助函数和类，用于支持主要的算法实现。这些工具函数可能包括：

1. **数据预处理函数**：用于处理和准备数据集，包括图像的缩放、归一化、增强等操作。

2. **性能评估函数**：计算不同的评价指标，如准确率、召回率、F1分数等，特别是针对小样本学习中常见的评估指标。

3. **模型保存和加载**：提供函数来保存和加载训练好的模型，方便后续使用或继续训练。

4. **超参数解析**：帮助处理命令行参数或配置文件中的超参数，使得算法更灵活和可配置。

5. **日志记录**：辅助记录训练过程中的日志信息，可能包括损失、精度和其他重要的训练指标。

6. **数据批量生成**：如果涉及到批量学习或在线学习，这里可能包含生成数据批次的函数。

7. **可视化工具**：提供一些函数或类用于绘制训练过程中的指标变化图、数据分布图等，帮助分析模型表现。

8. **距离和相似度计算**：在小样本学习中，经常需要计算样本间的距离或相似度，这里可能包含这样的实用函数。

这些功能有助于保持主算法的清晰和聚焦，同时提供必要的辅助功能和灵活性。
##### one-hot

这个函数将给定的索引`indices`转换为`one-hot`编码形式的张量，原来明确表示分类问题中的每个类别。

```example
indices = [2, 0, 1]
depth = 3
one-hot(indices,depth)
->
[[0, 0, 1],
 [1, 0, 0],
 [0, 1, 0]]
```
[数据预处理之独热编码](https://blog.csdn.net/zyc88888/article/details/103819604)
##### set_gpu

通过字符串`x`指定哪些GPU设备对CUDA可见，即哪些GPU设备可使用。

```example
x = '0,2'
->GPU0,2可用
```
##### ensure_path

这个函数的主要功能是确保指定的目录存在，如果目录已存在，则询问用户是否需要删除并重建。此外，它还可以将指定的脚本文件复制到该目录的`scripts`子目录中，以便保存实验或操作的相关脚本。这对于管理和追踪实验的不同版本非常有用。

##### Averager类

`Averager`类可以动态地计算和更新一组数值的平均值。你可以创建一个`Averager`实例，然后逐个使用`add`方法添加值，每次添加后都会自动更新平均值。当需要获取当前的平均值时，可以调用`item`方法。

这个类特别适合在需要逐步计算平均值的场景，例如在数据流处理或在线学习算法中

##### CrossEntropyLoss类

这个`CrossEntropyLoss`类实现了一个自定义的交叉熵损失，适用于多分类任务中。它首先将输入调整为合适的形状，然后计算对数`Softmax`，接着生成`one-hot`编码的目标张量，最后计算并返回平均交叉熵损失。这种实现方式为交叉熵损失的定制化提供了灵活性。
##### c_acc与count_acc

这两个函数本质上完成了相同的任务：计算分类准确率。它们都通过比较模型的预测类别和真实标签来确定预测正确的比例。主要区别在于它们处理张量和设备（CPU/GPU）的方式略有不同。

##### euclidean_metric

这个函数`euclidean_metric`计算了两组向量之间的欧几里得距离，并将其转换成了一个相似度分数。
##### Timer类

这段代码定义了一个名为`Timer`的类，用于测量从创建实例开始到调用`measure`方法的时间
##### pprint

可以很方便地在代码中使用pprint函数来获取数据结构的格式化输出，提高调试的效率和可读性。
##### compute_confidence_interval

这段代码定义了一个名为compute_confidence_interval的函数，它计算了一组数据的95%置信区间

函数返回两个值：m（样本均值）和pm（置信区间的半宽），因此95%置信区间可以表示为[m - pm, m + pm]
##### postprocess_args

这段代码定义了一个名为`postprocess_args`的函数，它处理并组织了一些输入参数，主要用于机器学习或深度学习中的实验设置。函数处理参数，构造路径，并最终更新参数对象。以下是详细的解释：

1. `def postprocess_args(args):`
   - 这行定义了一个名为`postprocess_args`的函数，它接受一个参数`args`，这通常是一个包含多个属性的对象。

2. `args.num_classes = args.way`
   - 这行代码将`args.way`的值赋给`args.num_classes`。在许多元学习和少样本学习的场景中，`way`表示任务中的类别数，因此这里将类别数设置为与`way`相同。

3. 构建`save_path1`：
   - 使用`'-'.join([...])`将几个参数组合成一个字符串，用作文件或目录的名称。
   - `args.dataset`、`args.model_class`、`args.backbone_class`是数据集名称、模型类别和模型的骨干网络类别。
   - `'{:02d}w{:02d}s{:02d}q'.format(args.way, args.shot, args.query)`格式化字符串，展示`way`（类别数）、`shot`（每类样本数）、`query`（查询样本数）的设置。

4. 构建`save_path2`：
   - 使用`'_'.join([...])`将另一组参数组合成字符串。
   - `' '.join(args.step_size.split(','))`处理学习率步长参数。
   - `'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul)`格式化学习率和其倍乘因子。
   - `'T1{}T2{}'.format(args.temperature, args.temperature2)`、`'b{}'.format(args.balance_1)`、`'bsz{:03d}'.format(...)`分别展示温度参数、平衡参数和批量大小。

5. 附加标志到`save_path1`：
   - `if args.init_weights is not None:`和`if args.use_euclidean:`根据是否初始化权重和是否使用欧几里得距离进行调整。

6. 附加标志到`save_path2`：
   - `if args.fix_BN:`和`if not args.augment:`根据是否固定BN层和是否应用数据增强进行调整。

7. 创建目录和最终保存路径：
   - `if not os.path.exists(os.path.join(args.save_dir, save_path1)):`检查目录是否存在，不存在则创建。
   - `args.save_path = os.path.join(args.save_dir, save_path1, save_path2)`设置最终的保存路径。

8. `return args`
   - 返回更新后的`args`对象。

这个函数在配置实验和保存结果时非常有用，它通过自动化命名和目录管理来帮助追踪不同实验的设置和结果。
##### get_command_line_parser

这段代码定义了一个函数 `get_command_line_parser`，它使用 `argparse` 库创建和配置一个命令行参数解析器。这种做法在很多Python程序中用于处理和解析命令行参数，尤其是在需要灵活配置实验或应用程序参数的情况下。下面是对这个函数和各个参数的具体解释：

###### 函数定义

- `def get_command_line_parser():`
  - 这个函数不接受任何参数，返回一个配置好的 `argparse.ArgumentParser` 对象。

###### 参数解析器初始化

- `parser = argparse.ArgumentParser()`
  - 创建一个 `ArgumentParser` 对象，这是用来管理命令行参数输入的主要工具。

###### 添加命令行参数

函数中的 `add_argument` 方法用来定义命令行参数，包括参数名、类型、默认值等。以下是一些关键参数的解释：

- `--max_epoch`：指定最大的训练周期数，默认是200。
- `--episodes_per_epoch`：每个训练周期包含的episode数，默认是100。
- `--num_eval_episodes`：评估阶段使用的episode数，默认是600。
- `--model_class`：模型类别，可选`INSTA_ProtoNet`和`ProtoNet`，默认是`INSTA_PorotNet`（可能是拼写错误，应该是`INSTA_ProtoNet`）。
- `--use_euclidean`：是否使用欧几里得距离，这是一个布尔标志，默认关闭。
- `--use_AdamW`：是否使用AdamW优化器，这是一个布尔标志，默认关闭。
- `--backbone_class`：模型的骨干网络类别，可选`Res12`和`Res18`，默认是`Res12`。
- `--dataset`：数据集名称，可选`MiniImageNet`、`TieredImageNet`、`CUB`、`FC100`，默认是`MiniImageNet`。
- `--way`、`--eval_way`：训练和评估时的分类方式（类别数），默认都是5。
- `--shot`、`--eval_shot`：训练和评估时的支持集样本数（每类），默认都是1。
- `--query`、`--eval_query`：训练和评估时的查询集样本数（每类），默认都是15。
- `--temperature`、`--temperature2`：在模型中使用的温度参数，用于调节某些算法的行为，两者默认都是1。

###### 优化和其他参数

- `--orig_imsize`：原始图像尺寸参数，特定数据集使用，默认是-1（无缓存）。
- `--lr`：学习率，默认是0.0001。
- `--lr_mul`：学习率乘数，默认是10。
- `--lr_scheduler`：学习率调度器，可选`multistep`、`step`、`cosine`，默认是`step`。
- `--step_size`：学习率步长，类型为字符串，默认是`20`。
- `--gamma`：学习率调整因子，默认是0.2。
- `--fix_BN`：是否固定批归一化中的运行均值/方差，默认关闭。
- `--augment`：是否应用数据增强，默认关闭。
- `--multi_gpu`：是否使用多GPU训练，默认关闭。
- `--gpu`：指定使用的GPU编号，默认是`0`。
- `--init_weights`：模型初始化权重文件路径，默认是None。
- `--testing`：是否为测试模式，默认关闭。

###### 常规但不常改动的参数

- `--mom`：动量值，默认是0.9。
- `--weight_decay`：权重衰减值，默认是0.0005。
- `--num_workers`：数据加载时使用的工作线程数，默认是8。
- `--log_interval`：日志记录间隔，默认是50。
- `--eval_interval`：评估间隔，默认是1。
- `--save_dir`：模型和日志保存目录，默认是`./checkpoints`。

###### 函数返回

- `return parser`
  - 返回配置好的 `argparse.ArgumentParser` 对象，供主程序或模块调用以解析命令行输入。

这个 `get_command_line_parser` 函数为程序提供了丰富的配置选项，使得训练和评估过程更加灵活和可调整。
