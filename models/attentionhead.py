import torch

import torch.nn as nn
import torch.nn.functional as F
from models.additive import AdditiveClassifier

def initialize_weights(m: torch.nn.Module):
    """Initialize fully connected layers with xavier initialization

    Parameters
    ----------
    m: torch.nn.Module\
        The torch module. If it's a linear layer, it will be initialized with xavier initialization and zeros for bias
        Otherwise, this function will do nothing
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class GatedAttention(nn.Module):
    """Perform gated attention operation

    Parameters
    ----------
    input_dim : int
        The number of input dimensions of the input. This corresponds to the number of output channels from the
        feature extractor/encoder.
    bottleneck_dim : int
        The bottleneck layers which squeezes the number of neurons input_dim to bottleneck_dim.
    dropout : bool
        Whether to include dropout in the model. When applied, it will drop 25% of connections
    num_branches: int
        The number of attention blocks to create, default is 1

    References
    ----------
    .. [*] Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        bottleneck_dim: int = 256,
        dropout: bool = False,
        n_branches: int = 1,
    ):
        super().__init__()
        self.attention_a = [nn.Linear(input_dim, bottleneck_dim), nn.Tanh()]
        self.attention_b = [nn.Linear(input_dim, bottleneck_dim), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(bottleneck_dim, n_branches)

    def forward(self, x):
        a = self.attention_a(x) #torch.Size([1, 625, 256]) #x torch.Size([1, 625, 512])
        b = self.attention_b(x) #torch.Size([1, 625, 256])
        att = a.mul(b) #torch.Size([1, 625, 256])
        att = self.attention_c(att)  # N x n_classes
        return att, x


class AttentionSingleBranch(nn.Module):
    """Single branch weakly supervised classification and segmentation with additive constraints.

    This class implements the single branch attention networks with additional additive constraints.
    The additive constraint allows the attention maps to be converted into a heatmap with a logit/probability
    interpretation, which gives more concise information about the contribution of each pixel towards the predicted
    class.

    If the additive model is used, an additional fully connected layer transforms the attention scores into n_classes
    additional heatmaps. Each of these heatmaps is tied to one specific class. We then perform a sum operation over
    each heatmap individually, which we will use as the image-level logit score. A softmax or sigmoid can then be
    used to calculate the final prediction for the image.

    Parameters
    ----------
    size: list[int, int, int] | None
        A list with 3 integers specifiying the sizes for the input dimensions for the attention network, the bottleneck
        dimension for the fully connected layers, and the bottleneck dimension within the attention network.
    use_dropout : bool
        Whether to use dropout layers within the network. If true, it will randomly disable 25% of the connections
    n_classes: int
        The number of classes of the weakly supervised classificatio
    additive: bool
        Whether to use additive classifiers. If true, an additional heatmap is created, equal to the amount of classes
        specified in n_classes.


    Symbols
    ----------
    C: The number of channels from the feature extractor
    K: Number of classes to predict
    N: The number of spatial elements in the feature map (height of the image, multiplied by the width of the image).
    B : The batch size. Usually 1 for pathology tasks

    References
    ----------
    . [*] Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning.
    . [*] Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo
           and Mahmood, Faisal. (2021). Data-efficient and weakly supervised computational pathology on whole-slide
           images. Nature Biomedical Engineering
    . [*] S.A. Javed, D. Juyal, H. Padigela, A. Taylor-Weiner, L. Yu, A. Prakash. (2022).
           Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology
    """

    def __init__(
        self,
        gate=None, #not used
        size_arg = "small", 
        #size: tuple[int, int, int] | None = None,
        # use_dropout: bool = False,
        dropout: float = 0.25,
        n_classes: int = 2,
        additive: bool = False,
        embed_dim=1024
    ):
        super().__init__()

        size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = size_dict[size_arg]

        self.additive = additive
        self.n_classes = n_classes

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        #if dropout:
        fc.append(nn.Dropout(dropout)) # Changed to using dropout probability instead of boolean flag to keep the layer consistent in eval and training
        attention_net = GatedAttention(input_dim=size[1], bottleneck_dim=size[2], dropout=dropout, n_branches=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)

        if self.additive:
            self.additive_classifiers = AdditiveClassifier(input_dims=size[1], output_dims=n_classes, hidden_dims=[])

        self.apply(initialize_weights)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.additive:
            self.additive_classifiers = self.additive_classifiers.to(device)
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def _compute_attention(self, x):
        """
        Compute the attention given a feature matrix x
        Parameters
        ----------
        x : torch.Tensor
            A [B,N,C] feature map consisting of N elements and C channels
        Returns
        -------
        x_transformed : torch.Tensor
            The transformed input matrix x (has fewer dimensions than the input x due to the fc layers).
        att_raw: torch.Tensor
            A [BxKxN] attention matrix.
        att: torch.Tensor:
            A [BxKxN] softmax-normalized attention matrix
        """
        att, x_transformed = self.attention_net(x)  # x_transformed = torch.Size([929, 512])
        att = torch.transpose(att, 1, 0)  # BxKxN  #att torch.Size([1, 1, 625])
        att_raw = att
        att = F.softmax(att, dim=1)  # softmax over N 
        return x_transformed, att_raw, att

    def forward(self, x):
        """
        Forward a feature vector through the attention network
        Parameters
        ----------
        x : torch.Tensor
            A [B, N, C] tensor containing the number of elements N in the feature map and the number of channels C
        Returns
        -------

        """
        x, att_raw, att = self._compute_attention(x) # x=h, att_raw=A_raw, att=A
        #pooled_features = torch.bmm(att, x) #batch matrix-matrix product # M = torch.mm(A, h) #torch.Size([1, 1, 512])
        pooled_features = torch.mm(att, x) #torch.Size([1, 512])
        att_pool_logits = self.classifiers(pooled_features)

        results_dict = {"attention": att_raw}

        if self.additive:
            classifier_out_dict = self.additive_classifiers(x, att.transpose(0, 1))
            bag_logits = classifier_out_dict["logits"]
            results_dict.update(
                {"patch_logits": classifier_out_dict["patch_logits"], "att_pool_logits": att_pool_logits} 
            )

            return bag_logits, att_raw, results_dict
        return att_pool_logits, att_raw, results_dict

class AttentionMultiBranch(AttentionSingleBranch):
    """Multi-branch weakly supervised classification and segmentation with additive constraints.

    Instead of using only 1 attention branch, this module will create a separate attention branch for each class
    from the start.

    This class implements the multi-branch attention networks with additional additive constraints.
    The additive constraint allows the attention maps to be converted into a heatmap with a logit/probability
    interpretation, which gives more concise information about the contribution of each pixel towards the predicted
    class.

    If the additive model is used, an additional fully connected layer transforms the n_classes attention scores
    into n_classes additional heatmaps. Each of these heatmaps is tied to one specific class. A sum operation over
    each heatmap individually is then performed, which we will use as the image-level logit score.
    A softmax or sigmoid can then be used to calculate the final prediction for the image.

    Parameters
    ----------
    size: list[int, int, int] | None
        A list with 3 integers specifiying the sizes for the input dimensions for the attention network, the bottleneck
        dimension for the fully connected layers, and the bottleneck dimension within the attention network.
    use_dropout : bool
        Whether to use dropout layers within the network. If true, it will randomly disable 25% of the connections
    n_classes: int
        The number of classes of the weakly supervised classificatio
    additive: bool
        Whether to use additive classifiers. If true, an additional heatmap is created, equal to the amount of classes
        specified in n_classes.

    References
    ----------
    .. [*] Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning.
    .. [*] Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo
           and Mahmood, Faisal. (2021). Data-efficient and weakly supervised computational pathology on whole-slide
           images. Nature Biomedical Engineering
    .. [*] S.A. Javed, D. Juyal, H. Padigela, A. Taylor-Weiner, L. Yu, A. Prakash. (2022).
           Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology
    """

    def __init__(
        self,
        #size: tuple[int, int, int] | None = None,
        size= None,
        dropout: float = 0.25,
        n_classes: int = 2,
        additive: bool = False,
        embed_dim=1024
    ):
        nn.Module.__init__(self)

        if size is None:
            size = (embed_dim, 512, 256)
        self.additive = additive
        self.n_classes = n_classes

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(dropout)) 

        attention_net = GatedAttention(
            input_dim=size[1],
            bottleneck_dim=size[2],
            dropout=dropout,
            n_branches=n_classes,
        )

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        if self.additive:
            self.additive_classifiers = nn.ModuleList(
                [AdditiveClassifier(input_dims=size[1], output_dims=1, hidden_dims=[]) for i in range(n_classes)]
            )

        # use an independent linear layer to predict each class
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]

        self.classifiers = nn.ModuleList(bag_classifiers)
        self.apply(initialize_weights)

    def forward(self, x):
        """
        Forward a feature vector through the attention network
        Parameters
        ----------
        x : torch.Tensor
            A [B, N, C] tensor containing the number of elements N in the feature map and the number of channels C
        Returns
        -------

        """
        device = x.device
        x, att_raw, att = self._compute_attention(x)  # att is [B, K, N]
        b, n, _ = x.shape
        att_pool_logits = torch.empty(b, self.n_classes).float().to(device)
        logits = torch.empty(b, self.n_classes).float().to(device)
        patch_logits = torch.zeros(b, n, self.n_classes).float().to(device)

        pooled_features = torch.bmm(att, x)
        for c in range(self.n_classes):
            att_pool_logits[:, c] = self.classifiers[c](pooled_features[:, c, :]).squeeze(1)

        results_dict = {"attention": att_raw}
        if self.additive:
            for c in range(self.n_classes):
                classifier_out_dict = self.additive_classifiers[c](x, att[:, c, :].unsqueeze(1).transpose(1, 2))
                logits[:, c] = classifier_out_dict["logits"].squeeze()
                patch_logits[..., c] = classifier_out_dict["patch_logits"].squeeze()

            results_dict.update({"patch_logits": patch_logits, "att_pool_logits": att_pool_logits})
            return logits, results_dict

        return att_pool_logits, results_dict


