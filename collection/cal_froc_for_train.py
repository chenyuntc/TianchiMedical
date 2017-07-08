%cd /home/x/dcsb/refactor
# 测试分类
%run -i test_class.py  doTest --model_dir='checkpoints/luna2016_0701_03:29:34.pth' --img_dir='/mnt/7/0630_train_no_normalization/' --csv_file='new_train_1.csv'
# 判断是否判断正确
%run -i collection/process.py check_nodule new_train_1.csv new_train_2.csv
# 对于每个节点的高概率进行加权
%run -i collection/process.py pcsv new_train_2.csv new_train_3.csv
# 计算froc
%run -i collection/cal_froc.py  main new_train_3.csv 1245 800

