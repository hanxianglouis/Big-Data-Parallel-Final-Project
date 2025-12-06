import torch
import torch.nn.functional as F

def local_prediction_loss(output: torch.Tensor, treatment: torch.Tensor, cost: torch.Tensor, revenue: torch.Tensor) :
    """
    Input: 
        output: the output of the model with size [batch_size,4]. Each column is a predicted value of cost or revenue under treated or untreated condition.
            1st column: treated cost;
            2nd column: treated revenue;
            3rd column: untreated cost;
            4th column: untreated revenue.
        treatment: the indicator of whether the user is treated or untreated in real world with size [batch_size].
        cost: the actual cost of the user, which is a binary variable.
        revenue: the actual revenue of the user, which is a binary variable.

    Output: the local prediction lost

    For one sample, if the treatement is 1, the loss will be half of the BCE Loss between 1st column of output and cost plus half of the BCE Loss between 2nd column of output and revenue.
    If the treatement is 0, the loss will be half of the BCE Loss between 3rd column of output and cost plus half of the BCE Loss between 4th column of output and revenue.
    """
    # masks
    treated_mask   = (treatment == 1)
    untreated_mask = (treatment == 0)

    # selects the correct tower output
    # cost
    pred_cost_treated   = output[treated_mask,   0]
    pred_cost_untreated = output[untreated_mask, 2]

    # revenue
    pred_rev_treated   = output[treated_mask,   1]
    pred_rev_untreated = output[untreated_mask, 3]

    # corresponding true labels
    true_cost_treated   = cost[treated_mask]
    true_cost_untreated = cost[untreated_mask]

    true_rev_treated   = revenue[treated_mask]
    true_rev_untreated = revenue[untreated_mask]

    # BCE with logits
    loss_cost_treated = F.binary_cross_entropy_with_logits(pred_cost_treated, true_cost_treated)
    loss_cost_untreated = F.binary_cross_entropy_with_logits(pred_cost_untreated, true_cost_untreated)

    loss_rev_treated = F.binary_cross_entropy_with_logits(pred_rev_treated, true_rev_treated)
    loss_rev_untreated = F.binary_cross_entropy_with_logits(pred_rev_untreated, true_rev_untreated)

    # aggregated losses
    L_c_local = loss_cost_treated + loss_cost_untreated
    L_r_local = loss_rev_treated + loss_rev_untreated

    # final supervised loss
    L_task = 0.5 * L_c_local + 0.5 * L_r_local

    return L_task

def decision_policy_learning_loss(output: torch.Tensor, treatment: torch.Tensor, cost: torch.Tensor, revenue: torch.Tensor, device: str) :
    """
    Input: 
        output: the output of the model with size [batch_size,4]. Each column is a predicted value of cost or revenue under treated or untreated condition.
            1st column: treated cost;
            2nd column: treated revenue;
            3rd column: untreated cost;
            4th column: untreated revenue.
        treatment: the indicator of whether the user is treated or untreated in real world with size [batch_size].
        cost: the actual cost of the user, which is a binary variable.
        revenue: the actual revenue of the user, which is a binary variable.

    Output: the decision policy learning lost
    """

    _lambda = 0.5
    # masks
    treated_mask   = (treatment == 1)
    untreated_mask = (treatment == 0)

    c1 = output[:,0]
    r1 = output[:,1]
    c0 = output[:,2]
    r0 = output[:,3]
    
    u1 = torch.exp(torch.clamp(r1,max=10))-_lambda * torch.exp(torch.clamp(c1,max=10))
    u0 = torch.exp(torch.clamp(r0,max=10))-_lambda * torch.exp(torch.clamp(c0,max=10))

    u = torch.stack([u0, u1], dim=1)  # concatenate u1 and u0, to shape [batch_size,2]

    p_all = F.softmax(u, dim=1)  # use softmax to calculate the probability

    p0 = p_all[:, 0]  # probability of choosing 0
    p1 = p_all[:, 1]  # probability of choosing 1

    p = torch.zeros(len(output),device=device) # the probability of choosing the real condition
    p[treated_mask] = p1[treated_mask]
    p[untreated_mask] = p0[untreated_mask]

    w1 = len(treatment) / treated_mask.sum()
    w0 = len(treatment) / untreated_mask.sum()

    w = torch.zeros(len(output),device=device)
    w[treated_mask] = w1
    w[untreated_mask] = w0

    r = revenue - _lambda * cost

    L_decision = (w * p * r).sum()
    return L_decision