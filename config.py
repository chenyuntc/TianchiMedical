#encoding:utf-8
import time
tfmt = '%m%d_%H%M%D'
class DefaultConfig:
    '''
    不要在这里修改
    '''
    data_root = '/home/x/data/dcsb/Tianchi' # 数据保存路径
    candidate_center="/home/x/dcsb/Tianchi_pytorch/csv/center.csv"#分割网络产生的疑似结点中心位置保存路径
    annotatiion_csv = "/home/x/dcsb/Tianchi_pytorch/csv/annotations.csv"
    information_csv='/home/x/dcsb/Tianchi_pytorch/csv/information.csv'#此csv文件保存了全部样本的原点以及spacing信息
    data_train='/home/x/data/datasets/tianchi/train/'#全部原始训练样本
    nodule_cubic='/mnt/7/train_nodule_cubic/'#从训练样本上切下的结点立方体保存路径
    candidate_cubic='/mnt/7/0705_train_48_64_candidate/'#从训练样本上切下的候选结点立方体保存路径
    save_file = time.strftime(tfmt+'.csv') # 保存文件尽量加上时间信息
    shuffle = True # 是否需要打乱数据
    num_workers = 4 # 多线程加载所需要的线程数目
    pin_memory =  True #数据从CPU->pin_memory—>GPU加速
    batch_size = 4
    ratio=5
    seg_ratio=2
    train_crop_size=48#训练分割网络时，切块的大小
    test_crop_size=[64,96,96]#测试分割网络时，切块的大小
    
    cls = False # False 代表进行分割，反之进行分类
    
    env = time.strftime(tfmt) # Visdom env
    plot_every = 10 # 每10个batch，更新visdom等
    weight_decay=1e-3

    max_epoch=100
    lr = 1e-3 # 学习率
    min_lr = 1e-7 # 当学习率低于这个值，就退出训练
    lr_decay = 0.8 # 当一个epoch的损失开始上升lr = lr*lr_decay 

    seg_model = 'Seg' # 模型，必须在models/__init__.py中import 
    seg_loss_function='dice_loss' #损失函数,对应于models.loss.py中的函数名
    seg_model_path=None # 预训练模型的路径
    seg_debug_file='/tmp/debug_seg' # 当该文件存在时，可能进入调试模式

    cls_model = 'Luna2016'
    # cls_model = 'MutltiCNN'
    # cls_loss_function='loss4multi' #损失函数,对应于models.loss.py中的函数名
    cls_loss_function='classifier_loss' #损失函数,对应于models.loss.py中的函数名
    cls_model_path=None #预训练模型的路径
    cls_debug_file='/tmp/debug_cls'
    
    optimizer_path='optimizer.pth' # 优化器的保存地址

class Config(DefaultConfig):
    '''
    在这里修改,覆盖默认值
    '''
opt = Config()