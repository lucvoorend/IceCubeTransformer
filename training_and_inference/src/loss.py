import torch
import scipy.special
import numpy as np

def MSE_loss(y_pred, target):
    y_pred = y_pred.squeeze()
    # Compute manual MSE loss
    loss = torch.mean((y_pred - target) ** 2)
    return loss


def dir_3vec_loss(pred, target):
    """
    Directional 3-vector loss function:
    - computes the loss between predicted and target 3-vectors by calculating the euclidean distance between points
    - returns the loss
    """
    x_pred, y_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    x_target, y_target, z_target = target[:, 0], target[:, 1], target[:, 2]

    loss = torch.sqrt((x_pred - x_target) ** 2 + (y_pred - y_target) ** 2 + (z_pred - z_target) ** 2)
    loss = torch.mean(loss)
    return loss

class LogCMK(torch.autograd.Function):
    """MIT License.

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indiciated in [1812.04616] in spite of suggestion in
    Sec. 8.2 of this paper. The change has been validated through comparison with
    exact calculations for `m=2` and `m=3` and found to yield the correct results.
    """

    @staticmethod
    def forward(
        self, 
        kappa,
        d = 3
    ):
        """Forward pass."""
        dtype = kappa.dtype
        self.save_for_backward(kappa)
        self.d = d
        self.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(d / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (d / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (d / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(
        self, grad_output
    ):
        """Backward pass."""
        kappa = self.saved_tensors[0]
        d = self.d
        dtype = self.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(d / 2.0, kappa))
            / (scipy.special.iv(d / 2.0 - 1, kappa))
        )
        return (
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
            None,
        )

def VonMisesFisherLoss3D(y_pred, target):
    """
    Von Mises Fisher loss function for 3D vectors

    Args:
    - y_pred (torch.Tensor): predicted 4D vector (x,y,z, kappa) [batch_size, 4]
    - target (torch.Tensor): target 3D vector (normalized) [batch_size, 3]

    Returns:
    - mean loss for the batch (torch.Tensor)
    """
    def log_cmk_exact(kappa, d=3):
        """
        Logarithm of the normalization constant of the von Mises Fisher distribution
        https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution

        WARNING: This function is not numerically stable for large values of kappa

        Args:
        - kappa (torch.Tensor): concentration parameter
        - d (int): dimension of the space (default=3)

        Returns:
        - log_cmk (torch.Tensor): logarithm of the normalization
        """
        return LogCMK.apply(kappa, d)
    
    def log_cmk_approx(kappa, d=3):
        """
        Logarithm of the normalization constant of the von Mises Fisher distribution
        https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution

        WARNING: This function is an approximation of cmk for large values of kappa
                 and gives wrong results for kappa < 100

        Args:
        - kappa (torch.Tensor): concentration parameter
        - d (int): dimension of the space (default=3)

        Returns:
        - log_cmk (torch.Tensor): logarithm of the normalization
        """
        v = d/2 - 0.5
        a = torch.sqrt((v+1)**2 + kappa**2)
        b = v - 1
        log_cmk_approx = -a + b * torch.log(b + a)
        return log_cmk_approx
    
    def log_cmk(kappa, d=3, switch=100):
        """
        Logarithm of the normalization constant of the von Mises Fisher distribution
        https://en.wikipedia.org/wiki/Von_Mises%E2%80%
        
        Automatically switches between exact and approximate calculation based on the
        value of kappa. The switch value is set to 100 by default.

        Args:
        - kappa (torch.Tensor): concentration parameter
        - d (int): dimension of the space (default=3)
        - switch (int): switch value between exact and approximate calculation (default=100)

        Returns:
        - log_cmk (torch.Tensor): logarithm of the normalization
        """
        switch = torch.tensor([switch]).to(kappa.device)
        mask_exact = kappa < switch
        
        # Calculate the offset between the exact and approximate calculation to ensure continuity
        offset = log_cmk_approx(switch, d) - log_cmk_exact(switch, d)

        log_cmk = log_cmk_approx(kappa, d)
        log_cmk[mask_exact] = log_cmk_exact(kappa[mask_exact], d)

        return log_cmk - offset

    kappa = y_pred[:, 3]
    # make sure kappa is positive
    kappa = torch.abs(kappa) + 1e-6

    p = kappa.unsqueeze(1) * y_pred[:, :3]
    norm_p = torch.norm(p, dim=1)

    dot_product = torch.sum(p * target, dim=1)

    log_cmk_val = log_cmk(norm_p)
    losses = -log_cmk_val - dot_product

    return torch.mean(losses)

def opening_angle_loss(y_pred, target):
    """
    Opening angle loss function:
    - computes the loss between predicted and target 3-vectors by calculating the opening angle between them
    - returns the loss
    """
    # Normalize the vectors
    y_pred = y_pred / torch.norm(y_pred, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    
    # Compute the dot product
    dot_product = torch.sum(y_pred * target, dim=1)

    # Ensure the dot product is in the range [-1, 1]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute the opening angle
    opening_angle = torch.acos(dot_product)
    
    # Compute the loss
    loss = torch.mean(opening_angle)
    return loss

def Simon_loss(y_pred, target):
    """"
    Simon loss function:
        - computes the loss between predicted and target 3-vectors by calculating the cosine similarity between them
        - uses kappa prediction
        - returns the loss
    """
    # Normalize the vectors
    pred_vec = y_pred[:, :3]
    target_vec = target

    # normalize the vectors
    pred_vec = pred_vec / torch.norm(pred_vec, dim=1).unsqueeze(1)
    target_vec = target_vec / torch.norm(target_vec, dim=1).unsqueeze(1)

    cosine_similarity = torch.sum(pred_vec * target_vec, dim=1)
    angle_loss = 1 - cosine_similarity

    lambda_kappa = 0.001  # Experiment with this weight
    epsilon = 1e-8  # Prevent log(0)

    zenith_pred = torch.arccos(y_pred[:, 2]/torch.norm(pred_vec, dim=1))
    azimuth_pred = torch.arctan2(y_pred[:, 1], y_pred[:, 0])

    log_kappa_zenith = torch.clamp(y_pred[:, 3], min=-2, max=2)
    log_kappa_azimuth = torch.clamp(y_pred[:, 4], min=-2, max=2)
    kappa_zenith = torch.exp(log_kappa_zenith)
    kappa_azimuth = torch.exp(log_kappa_azimuth)

    zenith_true = torch.arccos(target[:, 2]/torch.norm(target, dim=1))
    azimuth_true = torch.arctan2(target[:, 1], target[:, 0])

    kappa_loss = (-kappa_zenith * torch.cos(zenith_pred - zenith_true) + torch.log(kappa_zenith + epsilon)
                  - kappa_azimuth * torch.cos(azimuth_pred - azimuth_true) + torch.log(kappa_azimuth + epsilon))
    
    total_loss = angle_loss + lambda_kappa * torch.mean(kappa_loss, dim=0)

    total_loss = torch.mean(total_loss)
    return total_loss
