import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphConv (torch.nn.Module) :
    def __init__(
        self, 
        in_channels,                                                                                              # Input Channels
        out_channels,                                                                                             # Output Channels
        features_height,                                                                                          # Spatial height of features
        features_width,                                                                                           # Spatial width of features 
        edges,                                                                                                    # Number of edges in hypergraph convolution - A Hyperparamter
        filters=64,                                                                                               # Intermeditate channels for phi and lambda matrices - A Hyperparameter
        apply_bias=True,                                                                                          
        trainable=True, 
        name=None, 
        dtype=None, 
        dynamic=False,
        activation = None,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features_height = features_height
        self.features_width = features_width
        self.vertices = self.features_height * self.features_width
        self.edges = edges
        self.apply_bias = apply_bias
        self.trainable = trainable
        self.filters = filters

        # self.phi_conv = tf.keras.layers.Conv2D (self.filters, kernel_size=1, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal())
        self.phi_conv = torch.nn.Conv2d(in_channels = in_channels,
                                    out_channels = self.filters,
                                    kernel_size = 1,
                                    stride=1,
                                    padding='same'
                                    )
        # self.A_conv = tf.keras.layers.Conv2D (self.filters, kernel_size=1, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal())
        self.A_conv = torch.nn.Conv2d(in_channels = in_channels,
                                    out_channels = self.filters,
                                    kernel_size = 1,
                                    stride=1,
                                    padding='same'
                                    )
        # self.M_conv = tf.keras.layers.Conv2D (self.edges, kernel_size=7, strides=1, padding='same', kernel_initializer=tf.keras.initializers.glorot_normal ())
        self.M_conv = torch.nn.Conv2d(in_channels = in_channels,
                                    out_channels = self.edges,
                                    kernel_size = 7,
                                    stride=1,
                                    padding='same'
                                    )

        # Make a weight of size (input channels * output channels) for applying the hypergraph convolution
        # self.weight_2 = self.add_weight (
        #     name='Weight_2',
        #     shape=[self.in_channels, self.out_channels],
        #     dtype=tf.float32,
        #     initializer=tf.keras.initializers.glorot_normal(),
        #     trainable=self.trainable
        # )
        self.weight_2 = torch.nn.Parameter(data=torch.normal(0, 1, size=(self.in_channels, self.out_channels)), requires_grad=True)

        # If applying bias on the output features, make a weight of size (output channels) 
        if apply_bias :
            # self.bias_2 = self.add_weight (
            #     name='Bias_2',
            #     shape=[self.out_channels],
            #     dtype=tf.float32,
            #     initializer=tf.keras.initializers.glorot_normal(),
            #     trainable=self.trainable
            # )
            self.bias_2 = torch.nn.Parameter(data=torch.normal(0, 1, size=(self.out_channels,)), requires_grad=True)

        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
            elif activation == 'ReLU':
                self.activation = torch.nn.ReLU()
            elif activation == 'ELU':
                self.activation = torch.nn.ELU(alpha=1.0, inplace=False)
            else:
                raise NotImplementedError("Could not get activation {}".format(activation))

    def forward (self, x) :
        # Summary of the hypergraph convolution
        # x shape - self.features_height * self.features_width * self.in_channels
        # features - x
        # H = phi * A * phi.T * M
        # phi = conv2D (features)
        # A = tf.linalg.tensor_diag (conv2D (gloabalAveragePooling (features))
        # D = tf.linalg.tensor_diag (tf.math.reduce_sum (H, axis=1))
        # B = tf.linalg.tensor_diag (tf.math.reduce_sum (H, axis=0))
        # L = I - D^(-0.5) H B^(-1) H.T D^(-0.5)
        # out = L * features * self.weight_2 + self.bias_2
        iB, iC, iH, iW = x.size()
        # Phi Matrix
        phi = self.phi_conv(x)
        # phi = tf.reshape(phi, shape=(-1, self.vertices, self.filters))
        phi = phi.view(iB, self.filters, self.vertices)
        phi = phi.permute(0, 2, 1)
        
        # Lambda Matrix
        # A = tf.keras.layers.GlobalAveragePooling2D () (x)
        # A = tf.expand_dims (tf.expand_dims (A, axis=1), axis=1)
        # A = self.A_conv (A)
        # A = tf.linalg.diag (tf.squeeze (A))
        A = x.mean([2, 3]) # B x C
        A = A[:, :, None, None] # B x C x 1 x 1
        A = self.A_conv(A)
        sA = torch.squeeze(A, (2, 3))
        iB, iC = sA.size()
        A = torch.diag(sA.flatten())
        A = A.reshape(iB, iC, iB, iC).sum(-2)

        # Omega Matrix
        M = self.M_conv(x)
        # M = tf.reshape (M, shape=(-1, self.vertices, self.edges))
        M = M.view(iB, self.edges, self.vertices)
        M = M.permute(0, 2, 1)
        
        # Incidence matrix
        # H = | phi * lambda * phi.T * omega |
        # H = tf.matmul (phi, tf.matmul (A, tf.matmul (tf.transpose (phi, perm=[0, 2, 1]), M)))
        # H = tf.math.abs (H)
        H = torch.matmul(phi, torch.matmul(A, torch.matmul(torch.permute(phi, [0, 2, 1]), M)))
        H = torch.abs(H)
        
        # Degree matrix
        # D = tf.math.reduce_sum (H, axis=2)
        D = torch.sum(H, dim=2)

        # Mutlpying with the incidence matrix to ensure no matrix developed is of large size - (number of vertices * number of vertices)
        # D_H = tf.multiply (tf.expand_dims (tf.math.pow (D, -0.5), axis=-1), H)
        uD  = torch.unsqueeze(torch.pow(D + 1e-10, -0.5), dim = -1)
        D_H = torch.mul(uD, H)
        
        # Edge degree Matrix
        # B = tf.math.reduce_sum (H, axis=1)
        # B = tf.linalg.diag (tf.math.pow (B, -1))
        B = torch.sum(H, dim=1)
        B = torch.pow(B, -1)
        iB, iC = B.size()
        B = torch.diag(B.flatten())
        B = B.reshape(iB, iC, iB, iC).sum(-2)

        
        # Reshape the input features to apply the Hypergraph Convolution
        features = x.view(iB, self.in_channels, self.vertices)
        features = features.permute(0, 2, 1)
        
        # Hypergraph Convolution
        out = features - torch.matmul(D_H, torch.matmul(B, torch.matmul(torch.permute(D_H, [0, 2, 1]), features)))
        out = torch.matmul(out, self.weight_2)
        if self.apply_bias :
            out = out + self.bias_2

        # Reshape to output size
        out = out.view(iB, self.features_height, self.features_width, self.out_channels)
        out = out.permute(0, 3, 1, 2)
        if self.activation is not None:
            out = self.activation(out)
        return out