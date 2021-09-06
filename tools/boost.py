#!/usr/bin/env python
# encoding: utf-8
"""
@author: ambershek
@last modification: 2021/8/20
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch, BoostTrainer
from fastreid.utils.checkpoint import Checkpointer

#根据命令行得到配置信息cfg
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg



#在baseline AGW的基础上增加视角判别网络D继续训练
def main(args):
    cfg = setup(args)

    # Module 1 - Baseline
    #导入训练好的baseline AGW　model    
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = True
    model = BoostTrainer.build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    #输出baseline性能
    # res = BoostTrainer.test(cfg, model)
    # print("**********Baseline results**********")
    # print(res)
    # print()
    
    # Module 2 - Sampling
    trainer = BoostTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # 采样视角混淆样本
    # trainer.sampling(cfg, model)
    # FLAG
    # 用baseline采样好了，先注释掉
    # LATER
    # 也就是对比学习一直都在这些样本中做？样本集不断更新会不会更好？

    # # Module 3 - Boost
    # # boosting with adversarial training and contrastive learning
    # # 增加视角判别网络D，输入为AGW网络的backbone层输出
    # # 对抗训练backbone和D，使用新写的BoostTrainer(DefaultTrainer的子类)
    # # BoostTrainer中的成员数据_trainer为新写的ViewDisTrainer类

    # # TODO 
    # 导入上一模块采样的视角混淆样本作为boost训练的输入数据
    # 要换训练集（不是原来cfg中指定的input了）
    
    return trainer.train()
    return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )