def disparity_to_depth(disparity_image, baseline):
    unknown_disparity = disparity_image == float('inf')
    depth_image = baseline / (disparity_image + 1e-7)
    depth_image[unknown_disparity] = float('inf')

    return depth_image


def compute_relative_depth_error(estimated_disparity, ground_truth_disparity, baseline, depth_range=(5, 10)):
    """
    :param estimated_disparity: N x 5, (t, x, y, p, disp)
    :param ground_truth_disparity: HxW
    :param baseline: float
    :param depth_range: (min_depth, max_depth)
    :return: average relative depth error
    """
    ground_truth_depth = disparity_to_depth(ground_truth_disparity, baseline)

    errors = []
    for _, x, y, _, disp in estimated_disparity:
        x,y = int(x),int(y)
        depth_gt = ground_truth_depth[y, x]
        if depth_gt < depth_range[0] or depth_gt >= depth_range[1]:
            continue
        depth_ = baseline / (disp + 1e-7)
        errors.append(abs(depth_ - depth_gt) / depth_gt)
    return sum(errors) / len(errors)


if __name__ == '__main__':
    import numpy as np
    estimated_disparity = np.asarray([
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 2],
        [0, 1, 0, 0, 3],
        [0, 1, 1, 0, 4]
    ])
    ground_truth_disparity = np.asarray([
        [1, 2],
        [3, 4]
    ])
    # ground_truth_disparity = np.asarray([
    #     [1, 3],
    #     [4, 5]
    # ])

    print(compute_relative_depth_error(estimated_disparity, ground_truth_disparity, 26))
