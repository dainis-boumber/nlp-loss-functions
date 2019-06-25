class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight, reduce=False)
        
    @staticmethod
    def make_one_hot(labels, C=2):
        one_hot = w(torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_())
        target = one_hot.scatter_(1, labels.data, 1)

        target = w(Variable(target))

        return target
        
    def forward(self, input, target):
        loss = self.nll(input, target)
        
        one_hot = FocalLossMultiLabel.make_one_hot(target.unsqueeze(dim=1), input.size()[1])
        inv_probs = 1 - input.exp()
        focal_weights = (inv_probs * one_hot).sum(dim=1) ** self.gamma
        loss = loss * focal_weights
        
        return loss.mean()
