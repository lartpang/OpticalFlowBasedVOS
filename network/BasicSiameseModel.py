# -*- coding: utf-8 -*-
# @Time    : 2020/8/18
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : baseline.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
import torch

from module.BasicModules import BasicSegHead, BasicTranslayer
from utils.base_class import BaseModel


class BasicSiameseModel(BaseModel):
    def __init__(self, backbone_info):
        super(BasicSiameseModel, self).__init__(backbone_info)
        self.merge_translayer = BasicTranslayer(out_c=64)
        self.rgbf_decoder = BasicSegHead(mid_c=64)
        self.flow_decoder = BasicSegHead(mid_c=64)

        if self.pretrain_path:
            print(f"Loading the pretrained params from {self.pretrain_path}")
            self.load_state_dict(torch.load(self.pretrain_path))

    def forward(self, data):
        curr_data = torch.cat([data['curr_jpeg'], data['curr_flow']], dim=0)
        shared_en_feats = self.shared_encoder(curr_data)
        chunked_shared_en_feats = [xy.chunk(2, dim=0) for xy in shared_en_feats]

        if self.training:
            preds = self.forward_train(chunked_shared_en_feats)
        else:
            preds = self.forward_test(chunked_shared_en_feats)
        return preds

    def forward_train(self, chunked_shared_en_feats):
        trans_feats = self.merge_translayer([
            torch.cat([jpeg_en_feat + flow_en_feat, flow_en_feat], dim=0)
            for jpeg_en_feat, flow_en_feat in chunked_shared_en_feats
        ])
        rgbf_trans_feat, flow_trans_feat = list(zip(*[xy.chunk(2, dim=0) for xy in trans_feats]))
        seg_logits_for_rgbf = self.rgbf_decoder(rgbf_trans_feat)
        seg_logits_for_flow = self.flow_decoder(flow_trans_feat)

        return dict(curr_seg=seg_logits_for_rgbf, curr_seg_flow=seg_logits_for_flow)

    def forward_test(self, chunked_shared_en_feats):
        rgbf_trans_feat = self.merge_translayer([
            jpeg_en_feat + flow_en_feat
            for jpeg_en_feat, flow_en_feat in chunked_shared_en_feats
        ])
        seg_logits_for_rgbf = self.rgbf_decoder(rgbf_trans_feat)

        return dict(curr_seg=seg_logits_for_rgbf)


if __name__ == '__main__':
    jpeg = torch.randn(4, 3, 320, 320).cuda()
    flow = torch.randn(4, 3, 320, 320).cuda()
    model = BasicSiameseModel(
        dict(
            freeze_bn=False,
            backbone='resnet',
            # pretrain_path='/home/lart/Coding/CODProj/output'
            #               '/FPN_S384_BS8_E60_AMPy_LR0.004_LTpoly_OTsgdtrick_MSn_INFOheatmap/pth/state_final.pth',
            # stage='all',
            backbone_cfg=dict(depth=50)
        )
    ).cuda()
    # print(model(data=dict(image=jpeg, hs=hs))['seg'].shape)
    print(sum([x.numel() for x in model.parameters()]),
          set([x.type() for x in model.parameters()]))
