import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import torch
import numpy as np
from LAMDA_SSL.utils import class_status
from LAMDA_SSL.utils import one_hot
from LAMDA_SSL.Augmentation.Vision.Mixup import Mixup
from LAMDA_SSL.Augmentation.Vision.Rotate import Rotate
from LAMDA_SSL.utils import Bn_Controller
import LAMDA_SSL.Config.ReMixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy


class ReMixMatch(InductiveEstimator, DeepModelMixin, ClassifierMixin):
    def __init__(
        self,
        alpha=config.alpha,  # Mixup中beta分布的参数
        lambda_u=config.lambda_u,  # 无监督损失的权重
        T=config.T,  # 软标签的温度缩放参数
        num_classes=config.num_classes,  # 类别数量
        warmup=config.warmup,  # 模型训练的预热阶段设置
        p_target=config.p_target,  # 有标签数据的目标分布
        lambda_s=config.lambda_s,  # 基于混合前数据计算的无监督损失的权重
        lambda_rot=config.lambda_rot,  # 旋转角度分类损失的权重
        rotate_v_list=config.rotate_v_list,  # 旋转角度的列表
        mu=config.mu,  # 每次训练批次中无标签样本与有标签样本的比例
        ema_decay=config.ema_decay,  # EMA衰减率
        weight_decay=config.weight_decay,  # 权重衰减（L2正则化）参数
        epoch=config.epoch,  # 总训练轮数
        num_it_epoch=config.num_it_epoch,  # 每个epoch的迭代次数
        num_it_total=config.num_it_total,  # 总迭代次数
        eval_epoch=config.eval_epoch,  # 每多少个epoch进行一次评估
        eval_it=config.eval_it,  # 每多少次迭代进行一次评估
        device=config.device,  # 训练设备（CPU或GPU）
        train_dataset=config.train_dataset,  # 训练数据集
        labeled_dataset=config.labeled_dataset,  # 有标签的数据集
        unlabeled_dataset=config.unlabeled_dataset,  # 无标签的数据集
        valid_dataset=config.valid_dataset,  # 验证数据集
        test_dataset=config.test_dataset,  # 测试数据集
        train_dataloader=config.train_dataloader,  # 训练数据的dataloader
        labeled_dataloader=config.labeled_dataloader,  # 有标签数据的dataloader
        unlabeled_dataloader=config.unlabeled_dataloader,  # 无标签数据的dataloader
        valid_dataloader=config.valid_dataloader,  # 验证数据的dataloader
        test_dataloader=config.test_dataloader,  # 测试数据的dataloader
        train_sampler=config.train_sampler,  # 训练数据的采样器
        train_batch_sampler=config.train_batch_sampler,  # 训练数据的批次采样器
        valid_sampler=config.valid_sampler,  # 验证数据的采样器
        valid_batch_sampler=config.valid_batch_sampler,  # 验证数据的批次采样器
        test_sampler=config.test_sampler,  # 测试数据的采样器
        test_batch_sampler=config.test_batch_sampler,  # 测试数据的批次采样器
        labeled_sampler=config.labeled_sampler,  # 有标签数据的采样器
        unlabeled_sampler=config.unlabeled_sampler,  # 无标签数据的采样器
        labeled_batch_sampler=config.labeled_batch_sampler,  # 有标签数据的批次采样器
        unlabeled_batch_sampler=config.unlabeled_batch_sampler,  # 无标签数据的批次采样器
        augmentation=config.augmentation,  # 数据增强方法
        network=config.network,  # 模型的神经网络结构
        optimizer=config.optimizer,  # 优化器
        scheduler=config.scheduler,  # 学习率调度器
        evaluation=config.evaluation,  # 评估方法
        parallel=config.parallel,  # 是否启用并行训练
        file=config.file,  # 模型保存的文件路径
        verbose=config.verbose,  # 是否启用详细输出
    ):
        # >> 参数说明：
        # >> - lambda_u: 无监督损失的权重。
        # >> - T: 软标签的温度缩放参数。
        # >> - num_classes: 类别数量。
        # >> - alpha: Mixup中的beta分布参数。
        # >> - p_target: 有标签数据的目标分布。
        # >> - lambda_s: 基于混合前数据计算的无监督损失的权重。
        # >> - lambda_rot: 旋转角度分类损失的权重。
        # >> - rotate_v_list: 旋转角度的列表。

        # 初始化DeepModelMixin基类
        DeepModelMixin.__init__(
            self,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            augmentation=augmentation,
            network=network,
            train_sampler=train_sampler,
            train_batch_sampler=train_batch_sampler,
            valid_sampler=valid_sampler,
            valid_batch_sampler=valid_batch_sampler,
            test_sampler=test_sampler,
            test_batch_sampler=test_batch_sampler,
            labeled_dataset=labeled_dataset,
            unlabeled_dataset=unlabeled_dataset,
            labeled_dataloader=labeled_dataloader,
            unlabeled_dataloader=unlabeled_dataloader,
            labeled_sampler=labeled_sampler,
            unlabeled_sampler=unlabeled_sampler,
            labeled_batch_sampler=labeled_batch_sampler,
            unlabeled_batch_sampler=unlabeled_batch_sampler,
            epoch=epoch,
            num_it_epoch=num_it_epoch,
            num_it_total=num_it_total,
            eval_epoch=eval_epoch,
            eval_it=eval_it,
            mu=mu,
            weight_decay=weight_decay,
            ema_decay=ema_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            evaluation=evaluation,
            parallel=parallel,
            file=file,
            verbose=verbose,
        )

        self.ema_decay = ema_decay  # EMA衰减率
        self.lambda_u = lambda_u  # 无监督损失权重
        self.lambda_s = lambda_s  # 基于混合前数据的无监督损失权重
        self.lambda_rot = lambda_rot  # 旋转角度损失的权重
        self.weight_decay = weight_decay  # 权重衰减
        self.warmup = warmup  # 预热阶段
        self.T = T  # 温度缩放参数
        self.alpha = alpha  # Mixup中的beta分布参数
        self.num_classes = num_classes  # 类别数量
        self.rotate_v_list = rotate_v_list  # 旋转角度列表
        self.p_model = None  # 模型的当前分布
        self.p_target = p_target  # 目标分布
        self.bn_controller = Bn_Controller()  # 批归一化控制器
        self._estimator_type = ClassifierMixin._estimator_type  # 分类器类型标识

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(
            copy.copy(self.train_dataset.unlabeled_transform), dim=0, x=1
        )
        self._train_dataset.add_unlabeled_transform(
            copy.copy(self.train_dataset.unlabeled_transform), dim=0, x=2
        )
        self._train_dataset.add_transform(self.weak_augmentation, dim=1, x=0, y=0)
        self._train_dataset.add_unlabeled_transform(
            self.weak_augmentation, dim=1, x=0, y=0
        )
        self._train_dataset.add_unlabeled_transform(
            self.strong_augmentation, dim=1, x=1, y=0
        )
        self._train_dataset.add_unlabeled_transform(
            self.strong_augmentation, dim=1, x=2, y=0
        )

    def start_fit(self):
        self.num_classes = (
            self.num_classes
            if self.num_classes is not None
            else class_status(self._train_dataset.labeled_dataset.y).num_classes
        )
        if self.p_target is None:
            class_counts = torch.Tensor(
                class_status(self._train_dataset.labeled_dataset.y).class_counts
            ).to(self.device)
            self.p_target = class_counts / class_counts.sum(dim=-1, keepdim=True)
        self._network.zero_grad()
        self._network.train()

    def train(self, lb_X, lb_y, ulb_X, lb_idx=None, ulb_idx=None, *args, **kwargs):
        # 如果lb_X或lb_y是元组或列表类型，则只取第一个元素
        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y

        # 将无标签数据ulb_X解包为w_ulb_X, s_ulb_X_1, s_ulb_X_2三部分
        w_ulb_X, s_ulb_X_1, s_ulb_X_2 = ulb_X[0], ulb_X[1], ulb_X[2]

        # 初始化一个空张量s_ulb_rot_X_1用于存储旋转后的图像，并初始化旋转角度列表
        s_ulb_rot_X_1 = torch.Tensor().to(self.device)
        rot_index = []

        # 对每个无标签样本的第一个版本应用随机旋转，并将旋转后的图像存入s_ulb_rot_X_1
        for item in s_ulb_X_1:
            _v = np.random.choice(self.rotate_v_list, 1).item()  # 随机选择一个旋转角度
            s_ulb_rot_X_1 = torch.cat(
                (s_ulb_rot_X_1, Rotate(v=_v).fit_transform(item).unsqueeze(0)), dim=0
            )  # 旋转图像并添加到s_ulb_rot_X_1
            rot_index.append(self.rotate_v_list.index(_v))  # 记录旋转角度的索引

        # 将旋转角度的索引转换为张量，放入设备
        rot_index = torch.LongTensor(rot_index).to(self.device)

        # 获取有标签数据的样本数量
        num_lb = lb_X.shape[0]

        with torch.no_grad():
            # 冻结Batch Normalization层的参数
            self.bn_controller.freeze_bn(model=self._network)

            # 使用网络对无标签数据的第一个版本进行前向传递，计算logits
            w_ulb_logits = self._network(w_ulb_X)[0]

            # 解冻Batch Normalization层
            self.bn_controller.unfreeze_bn(model=self._network)

            # 计算无标签数据的类别概率分布
            ulb_prob = torch.softmax(w_ulb_logits, dim=1)

            # 如果p_model还没有初始化，则初始化为ulb_prob的均值
            if self.p_model is None:
                self.p_model = torch.mean(ulb_prob.detach(), dim=0).to(self.device)
            else:
                # 更新p_model，使用指数滑动平均（EMA）更新策略
                self.p_model = (
                    self.p_model * 0.999 + torch.mean(ulb_prob.detach(), dim=0) * 0.001
                )

            # 调整无标签数据的概率分布，使其与p_target对齐
            ulb_prob = ulb_prob * self.p_target / self.p_model
            ulb_prob = ulb_prob / ulb_prob.sum(dim=-1, keepdim=True)

            # 对概率进行sharpen处理，温度缩放
            ulb_sharpen_prob = ulb_prob ** (1 / self.T)
            ulb_sharpen_prob = (
                ulb_sharpen_prob / ulb_sharpen_prob.sum(dim=-1, keepdim=True)
            ).detach()

            # 将有标签数据和无标签数据（包括其增强版本）拼接起来作为混合输入
            mixed_inputs = torch.cat((lb_X, s_ulb_X_1, s_ulb_X_2, w_ulb_X))

            # 为混合输入生成标签，包括有标签数据的真实标签和无标签数据的伪标签
            input_labels = torch.cat(
                [
                    one_hot(lb_y, self.num_classes, self.device).to(
                        self.device
                    ),  # 有标签数据的one-hot编码
                    ulb_sharpen_prob,  # 无标签数据的伪标签（增强数据1）
                    ulb_sharpen_prob,  # 无标签数据的伪标签（增强数据2）
                    ulb_sharpen_prob,  # 无标签数据的伪标签（原始数据）
                ],
                dim=0,
            )

            # 随机打乱混合输入的顺序
            index = torch.randperm(mixed_inputs.size(0)).to(self.device)

            # 使用Mixup技术对输入和标签进行混合
            mixed_X, mixed_y = (
                Mixup(self.alpha)
                .fit((mixed_inputs, input_labels))
                .transform((mixed_inputs[index], input_labels[index]))
            )

            # 将混合后的输入分割成有标签数据和无标签数据的部分
            mixed_X = list(torch.split(mixed_X, num_lb))
            mixed_X = self.interleave(mixed_X, num_lb)

        # 对有标签数据的混合输入进行前向传递
        _mix_0 = self._network(mixed_X[0])[0]
        logits = [_mix_0]

        # 冻结Batch Normalization层，避免统计信息的干扰
        self.bn_controller.freeze_bn(model=self._network)

        # 对其余无标签数据的混合输入进行前向传递
        for ipt in mixed_X[1:]:
            _mix_i = self._network(ipt)[0]
            logits.append(_mix_i)

        # 对增强后的无标签数据（版本1）计算logits
        s_ulb_logits_1 = self._network(s_ulb_X_1)[0]

        # 对旋转后的无标签数据计算旋转预测的logits
        rot_logits = self._network(s_ulb_rot_X_1)[1]

        # 将logits重新组合
        logits = self.interleave(logits, num_lb)

        # 解冻Batch Normalization层
        self.bn_controller.unfreeze_bn(model=self._network)

        # 从logits中分割出有标签数据的logits和无标签数据的logits
        lb_logits = logits[0]
        ulb_logits = torch.cat(logits[1:])

        # 返回所有所需的值用于后续计算损失
        return (
            lb_logits,  # 有标签数据的logits
            mixed_y[:num_lb],  # 有标签数据的混合标签
            ulb_logits,  # 无标签数据的logits
            mixed_y[num_lb:],  # 无标签数据的混合标签
            s_ulb_logits_1,  # 增强数据1的logits
            ulb_sharpen_prob,  # 无标签数据的sharpen伪标签
            rot_logits,  # 旋转预测的logits
            rot_index,  # 旋转角度的索引
        )

    def interleave_offsets(self, batch, num):
        groups = [batch // (num + 1)] * (num + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        num = len(xy) - 1
        offsets = self.interleave_offsets(batch, num)
        xy = [[v[offsets[p] : offsets[p + 1]] for p in range(num + 1)] for v in xy]
        for i in range(1, num + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def get_loss(self, train_result, *args, **kwargs):
        (
            lb_logits,
            mix_lb_y,
            ulb_logits,
            mix_ulb_y,
            s_ulb_logits_1,
            ulb_sharpen_prob,
            rot_logits,
            rot_index,
        ) = train_result
        sup_loss = Cross_Entropy(use_hard_labels=False, reduction="mean")(
            lb_logits, mix_lb_y
        )  # CE_loss for labeled data
        _warmup = float(
            np.clip((self.it_total) / (self.warmup * self.num_it_total), 0.0, 1.0)
        )
        unsup_loss = _warmup * Cross_Entropy(use_hard_labels=False, reduction="mean")(
            ulb_logits, mix_ulb_y
        )
        s_loss = _warmup * Cross_Entropy(use_hard_labels=False, reduction="mean")(
            s_ulb_logits_1, ulb_sharpen_prob
        )
        rot_loss = Cross_Entropy(reduction="mean")(rot_logits, rot_index)
        loss = (
            sup_loss
            + self.lambda_u * unsup_loss
            + self.lambda_s * s_loss
            + self.lambda_rot * rot_loss
        )
        return loss

    def predict(self, X=None, valid=None):
        return DeepModelMixin.predict(self, X=X, valid=valid)
