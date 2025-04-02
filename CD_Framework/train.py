import os
import torch
import argparse
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from eval import eval_for_metric
from losses.get_losses import SelectLoss
from models.block.Drop import dropblock_step
from util.dataloaders import get_loaders
from util.common import check_dirs, init_seed, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
from main_model import ChangeDetection, ModelEMA, ModelSWA


def train(opt):
    init_seed() # 设置随机种子
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()  # 打印GPU信息
    save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path = check_dirs()

    save_results = SaveResult(result_save_path) # 保存结果的类:SaveResult()--uti.common.py
    save_results.prepare() # 创造result.txt的日志文件

    train_loader, val_loader = get_loaders(opt) # 训练集,验证集导入数据的函数:get_loaders()--util.dataloaders.py ; 返回数据,传进训练迭代
    scale = ScaleInOutput(opt.input_size)

    model = ChangeDetection(opt).cuda() # 进入变化检测网络
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    criterion = SelectLoss(opt.loss) # 调用损失函数的类SelectLoss()--losses.get_losses.py

    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10},  
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}]  
        print("Using finetune for model")
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.001)
    if opt.pseudo_label:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate/5, epochs=opt.epochs, up_rate=0)
    else:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs)  

    best_metric = 0
    train_avg_loss = 0
    total_bs = 16
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))

    # 一次训练完成,开启下一轮训练,开始迭代
    for epoch in range(opt.epochs):
        model.train()
        train_tbar = tqdm(train_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(train_tbar):
            train_tbar.set_description("epoch {}, train_loss {}".format(epoch, train_avg_loss))
            if epoch == 0 and i < 20:
                save_results.save_first_batch(batch_img1, batch_img2, batch_label1, batch_label2, i)
            if opt.pseudo_label and epoch == 0:
                print("---Using Pseudo labels, skip the first epoch!---")
                break

            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label1 = batch_label1.long().cuda()
            batch_label2 = batch_label2.long().cuda()

            batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))   # 指定某个尺度进行训练
            # print( batch_img1.shape)
            outs = model(batch_img1, batch_img2)
            # print( outs[0].shape)
            outs = scale.scale_output(outs)
            # print( batch_label2.shape)
         
            # 利用构建的criterion损失函数 ---> 与标签图比较计算损失 ---> 反向传播
            loss = criterion(outs, (batch_label1, batch_label2)) if model.dl else criterion(outs, (batch_label1,))
            train_avg_loss = (train_avg_loss * i + loss.cpu().detach().numpy()) / (i + 1)
            # 反向传播
            loss.backward()
            if ((i+1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()

            del batch_img1, batch_img2, batch_label1, batch_label2

        scheduler.step()
        dropblock_step(model)
        # swa.update(model)
        # swa.save(every_ckp_save_path)

        # 指标计算 ---> 模型优化
        p, r, f1, miou, oa, val_avg_loss = eval_for_metric(model, val_loader, criterion, input_size=opt.input_size)

        refer_metric = f1
        #refer_metric = miou
        underscore = "_"
        # 模型优化 ---> 记录最好的结果
        if refer_metric.mean() > best_metric:
            if best_ckp_file is not None:
                os.remove(best_ckp_file)
            best_ckp_file = os.path.join(
                best_ckp_save_path,
                underscore.join([opt.backbone, opt.neck, opt.head, 'epoch',
                                 str(epoch), str(round(float(refer_metric.mean()), 5))]) + ".pt")
            torch.save(model, best_ckp_file)
            best_metric = refer_metric.mean()
                
        # 写日志
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        save_results.show(p, r, f1, miou, oa, refer_metric, best_metric, train_avg_loss, val_avg_loss, lr, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')

    # 配置模型
    parser.add_argument("--backbone", type=str, default="focalnet_96") # backbone_调用backbone里模型的通道数(embed_dim) # backbone常为网络中的encoder（编码器）
    parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop") # 特征融合阶段
    parser.add_argument("--head", type=str, default="fcn") # 预测头
    parser.add_argument("--loss", type=str, default="bce+dice") # 损失计算阶段

    # 配置训练参数
    parser.add_argument("--pretrain", type=str,
                        default="")  # 预训练权重路径，在原有网络权重训练基础上进行训练
    # parser.add_argument("--pretrain", type=str, default="./runs/train/11/best_ckp/tf_efficientnetv2_s_in21k_w40_fpn+aspp+fuse+drop_fcn_epoch_81_0.91389.pt")  # 历史预训练权重路径1
    # parser.add_argument("--pretrain", type=str, default="./runs/train/1/best_ckp/focalnet_96_fpn+aspp+fuse+drop_fcn_epoch_346_0.92164.pt")  # 历史预训练权重路径2
    
    parser.add_argument("--cuda", type=str, default="2") # GPU编号,本地0/1卡
    parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk1/BCD_Dataset/LEVIR-CD/") # 数据集路径
    parser.add_argument("--batch-size", type=int, default=32) # "每批次处理"的数据量;在内存效率和内存容量之间寻找最佳平衡;较小的batch_size可能有助于模型泛化，而较大的batch_size可能导致过拟合;服务器中常见跑到70GB显存;
    parser.add_argument("--epochs", type=int, default=300)  # 训练轮数:当一个完整的数据集通过神经网络一次并且返回一次的过程称为一个epoch
    parser.add_argument("--input-size", type=int, default=224) # 数据集图片的尺寸
    parser.add_argument("--num-workers", type=int, default=2) # "并行(多线程)"数据数量
    parser.add_argument("--learning-rate", type=float, default=0.00035) # 学习率,基本不改
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=False)

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    train(opt)
