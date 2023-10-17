import torch
import torch.nn.functional as F


def color_constancy_loss(x: torch.Tensor):
    """
    Correct the potential color deviations in the enhanced image.
    """
    mean_rgb = torch.mean(x, axis=(2, 3), keepdims=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    drg = torch.pow(mr - mg, 2)
    drb = torch.pow(mr - mb, 2)
    dgb = torch.pow(mb - mg, 2)
    k = torch.sqrt(torch.pow(drg, 2) + torch.pow(drb, 2) + torch.pow(dgb, 2))
    return k


def illumination_smoothness_loss(x: torch.Tensor, loss_weight: float = 1.0):
    """
    To preserve the monotonicity relations between neighboring pixels, the illumination smoothness loss is added to each curve parameter map.
    """
    batch_size, _, h_x, w_x = x.shape
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    h_tv = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2))
    w_tv = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2))
    return loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def spatial_consistency_loss(x_enhanced: torch.Tensor, x_original: torch.Tensor, dev: str = 'cuda'):
    """
    Encourages spatial coherence of the enhanced image by preserving the contrast between neighboring regions across the input image and its enhanced version.
    """
    left_neighbour = torch.tensor(
        [[[
            [ 0.,  0.,  0.],
            [-1.,  1.,  0.],
            [ 0.,  0.,  0.]
        ]]],
        dtype=torch.float,
        device=dev
    )
    right_neighbour = torch.tensor(
        [[[
            [ 0.,  0.,  0.],
            [ 0.,  1., -1.],
            [ 0.,  0.,  0.]
        ]]],
        dtype=torch.float,
        device=dev
    )
    top_neighbour = torch.tensor(
        [[[
            [ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  0.]
        ]]],
        dtype=torch.float,
        device=dev
    )
    bottom_neighbour = torch.tensor(
        [[[
            [ 0.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0., -1.,  0.]
        ]]],
        dtype=torch.float,
        device=dev
    )

    original_pool = F.avg_pool2d(
        x_original, (4, 4), stride=1
    )
    enhanced_pool = F.avg_pool2d(
        x_enhanced, (4, 4), stride=1
    )
    
    left_neighbour = left_neighbour.expand(-1, original_pool.shape[1], -1, -1)
    right_neighbour = right_neighbour.expand(-1, original_pool.shape[1], -1, -1)
    top_neighbour = top_neighbour.expand(-1, original_pool.shape[1], -1, -1)
    bottom_neighbour = bottom_neighbour.expand(-1, original_pool.shape[1], -1, -1)

    d_org_left    = F.conv2d(original_pool, left_neighbour, padding=1)
    d_org_right   = F.conv2d(original_pool, right_neighbour, padding=1)
    d_org_top     = F.conv2d(original_pool, top_neighbour, padding=1)
    d_org_bottom  = F.conv2d(original_pool, bottom_neighbour, padding=1)

    d_enhance_left   = F.conv2d(enhanced_pool, left_neighbour, padding=1)
    d_enhance_right  = F.conv2d(enhanced_pool, right_neighbour, padding=1)
    d_enhance_top    = F.conv2d(enhanced_pool, top_neighbour, padding=1)
    d_enhance_bottom = F.conv2d(enhanced_pool, bottom_neighbour, padding=1)

    d_left   = torch.pow(d_org_left - d_enhance_left, 2)
    d_right  = torch.pow(d_org_right - d_enhance_right, 2)
    d_top    = torch.pow(d_org_top - d_enhance_top, 2)
    d_bottom = torch.pow(d_org_bottom - d_enhance_bottom, 2)
    return d_left + d_right + d_top + d_bottom


def exposure_control_loss(x: torch.Tensor):
    """
    Measures the distance between the average intensity value of a local region and a preset well-exposedness level (set to 0.6).
    """
    E = torch.tensor(0.6, dtype=torch.float)
    kernel = (16, 16)
    x = torch.mean(x, 1, keepdim=True)
    mean = F.avg_pool2d(x, kernel_size=kernel)
    return torch.mean(torch.pow(mean - E, 2))
