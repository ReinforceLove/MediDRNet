import argparse

import torch
import torch.nn as nn


class ProSupConLoss(nn.Module):

    def __init__(self, args, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, ):
        super(ProSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.Prototypical = torch.autograd.Variable(torch.randn(args.classes, args.contrast_fea_size).to(args.device),
                                                    requires_grad=True)

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1)
        # contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # labels = labels.repeat(contrast_count)
        contrast_feature = features
        # 计算余弦相似度
        anchor_dot_contrast = torch.div(
            self._cosine_simililarity(contrast_feature, self.Prototypical),
            self.temperature)

        # 计算logits
        logits = torch.exp(anchor_dot_contrast)

        # 计算每个样本的正负logits和
        pos = logits.gather(1, labels.unsqueeze(1)).view(-1)  # 选择labels对应的logits作为正样本
        neg = torch.sum(logits, dim=1) - pos  # 所有logits之和减去正样本的logits

        # 计算损失
        psc = -torch.log(pos / neg)
        # print(psc)
        # 计算总损失
        losses = torch.mean(psc)

        return torch.abs(losses)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="data/ACDC/train/unlabel")
    parser.add_argument("--contrast_fea_size", type=int, default=512)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--save', metavar='SAVE', default='./saves', help='saved folder')
    parser.add_argument('--model_save_name', type=str, help='Saving path', default='con.pt')
    args = parser.parse_args()

    x = torch.ones(8, 2, 512).cuda()
    y = torch.randint(0, 5, [8]).cuda()
    loss = ProSupConLoss(args).cuda()
    print(loss(x, labels=y))
