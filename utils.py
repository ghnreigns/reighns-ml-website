import torch


def torch_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The converted numpy array.
    """
    return tensor.detach().cpu().numpy()


def compare_equality_two_tensors(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> bool:
    """Compare two PyTorch tensors for equality.

    Args:
        tensor1 (torch.Tensor): The first PyTorch tensor to compare.
        tensor2 (torch.Tensor): The second PyTorch tensor to compare.

    Returns:
        bool: Whether the two tensors are equal.
    """
    if torch.all(torch.eq(tensor1, tensor2)):
        return True
    return False


def compare_closeness_two_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    epsilon: float,
    *args,
    **kwargs
) -> bool:
    """Compare two PyTorch tensors for closeness.

    Args:
        tensor1 (torch.Tensor): The first PyTorch tensor to compare.
        tensor2 (torch.Tensor): The second PyTorch tensor to compare.
        epsilon (float): The epsilon value to use for closeness.

    Returns:
        bool: Whether the two tensors are close.
    """
    if torch.allclose(tensor1, tensor2, atol=epsilon, *args, **kwargs):
        return True
    return False


z_logits = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=torch.float32)

y_true = torch.tensor([0, 2], dtype=torch.long)

y_true_ohe = torch.tensor([[1, 0, 0], [0, 0, 1]], dtype=torch.long)
assert y_true_ohe == torch.nn.functional.one_hot(y_true, num_classes=3)


def compute_softargmax(z: torch.Tensor) -> torch.Tensor:
    """Compute the softargmax of a PyTorch tensor.

    Args:
        z (torch.Tensor): The PyTorch tensor to compute the softargmax of.

    Returns:
        torch.Tensor: The softargmax of the PyTorch tensor.
    """

    # the output matrix should be the same size as the input matrix
    z_softargmax = torch.zeros(size=z.size(), dtype=torch.float32)

    for row_index, each_row in enumerate(z):
        denominator = torch.sum(torch.exp(each_row))
        for element_index, each_element in enumerate(each_row):
            z_softargmax[row_index, element_index] = (
                torch.exp(each_element) / denominator
            )

    assert compare_closeness_two_tensors(
        z_softargmax, torch.nn.Softmax(dim=1)(z), 1e-6
    )
    return z_softargmax


def compute_categorical_cross_entropy_loss(
    y_true: torch.Tensor, y_prob: torch.Tensor
) -> torch.Tensor:
    """Compute the categorical cross entropy loss between two PyTorch tensors.

    Args:
        y_true (torch.Tensor): The true labels in one-hot form.
        y_prob (torch.Tensor): The predicted labels in one-hot form.

    Returns:
        torch.Tensor: The categorical cross entropy loss.
    """

    all_samples_loss = 0
    for each_y_true_one_hot_vector, each_y_prob_one_hot_vector in zip(
        y_true, y_prob
    ):
        current_sample_loss = 0
        for each_y_true_element, each_y_prob_element in zip(
            each_y_true_one_hot_vector, each_y_prob_one_hot_vector
        ):
            # Indicator Function
            if each_y_true_element == 1:
                current_sample_loss += -1 * torch.log(each_y_prob_element)
            else:
                current_sample_loss += 0

        all_samples_loss += current_sample_loss

    all_samples_average_loss = all_samples_loss / y_true.shape[0]
    return all_samples_average_loss


def compute_categorical_cross_entropy_loss_dot_product(
    y_true: torch.Tensor, y_prob: torch.Tensor
) -> torch.Tensor:
    """Compute the categorical cross entropy loss between two PyTorch tensors using dot product.

    Args:
        y_true (torch.Tensor): The true labels in one-hot form.
        y_prob (torch.Tensor): The predicted labels in one-hot form.

    Returns:
        torch.Tensor: The categorical cross entropy loss.
    """
    m = torch.matmul(y_true.float(), torch.neg(torch.transpose(y_prob.float())))
    all_loss_vector = torch.diagonal(m, 0)
    all_loss_sum = torch.sum(all_loss_vector, dim=0)
    average_loss = all_loss_sum / y_true.shape[0]
    return average_loss
