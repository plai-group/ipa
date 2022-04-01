import torch
import numpy as np
import torchvision.utils as vutils
from utils import img_mask_to_np_input

def quantitative(model, dataloader, device, K=10):
        """
        Evaluates the model on the test set and computes a set of performance measures
        NLL loss: negative log likelihood on the unobserved target pixels
        l2 loss: average L2 distance between the groundtruth and prediction on the unobserved target pixels

        Args:
            dataloader (torch.utils.DataLoader):
                test loader or validation loader
        """
        max_k_local = 10
        model.eval()
        with torch.no_grad():
            nll_list, l2_list = [], []
            unobserved_nll_list, unobserved_l2_list = [], []
            for data in dataloader:
                # Create context and target points and apply neural process
                # The following part splits K into batches of at most k_local elements to avoid GPU out of memory
                k_local_list = np.arange(0, K, max_k_local, dtype=int)
                if k_local_list[-1] != K:
                    k_local_list = np.append(k_local_list, K)
                k_local_list = k_local_list[1:] - k_local_list[:-1]
                log_prob_parts, l2_parts = [], []
                x_context, y_context, x_target, y_target = data[0]
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)
                for k_local in k_local_list:
                    batch_size = len(x_context)
                    # Expand the img and masks to simulate doing inference multiple times
                    x_context_local = x_context.unsqueeze(0).repeat(k_local, *([1] * x_context.ndim)).reshape(k_local*batch_size, *x_context.shape[1:])
                    y_context_local = y_context.unsqueeze(0).repeat(k_local, *([1] * y_context.ndim)).reshape(k_local*batch_size, *y_context.shape[1:])
                    x_target_local = x_target.unsqueeze(0).repeat(k_local, *([1] * x_target.ndim)).reshape(k_local*batch_size, *x_target.shape[1:])
                    y_target_local = y_target.unsqueeze(0).repeat(k_local, *([1] * y_target.ndim)).reshape(k_local*batch_size, *y_target.shape[1:])
                    # Forward pass
                    p_y_pred = model.forward_test(x_context_local, y_context_local, x_target_local)
                    # Compute log_prob and add it to the list
                    log_prob_flat = p_y_pred.log_prob(y_target_local)
                    log_prob = log_prob_flat.view(k_local, batch_size, *log_prob_flat.shape[1:])
                    log_prob = log_prob.sum(dim=-1)
                    log_prob_parts.append(log_prob)
                    l2_parts.append(torch.sqrt((p_y_pred.loc - y_target_local)**2).mean(list(range(1, y_target_local.ndim))))
                
                log_prob = torch.cat(log_prob_parts, dim=0)
                assert K == log_prob.size(0)
                nll_list.append(-(log_prob.logsumexp(dim=0) - np.log(K)).mean().item())
                l2_list.append(torch.mean(torch.cat(l2_parts, dim=0)).item())

                # Compute validation error on unobserved part of target sets
                if data[1] is not None:
                    images, context_mask, target_mask = data[1]
                    target_mask = target_mask & (~context_mask)
                    if target_mask.sum() > 0:
                        log_prob_parts, l2_parts = [], []
                        x_context, y_context = img_mask_to_np_input(images, context_mask)
                        x_target, y_target = img_mask_to_np_input(images, target_mask)
                        x_context = x_context.to(device)
                        y_context = y_context.to(device)
                        x_target = x_target.to(device)
                        y_target = y_target.to(device)
                        for k_local in k_local_list:
                            # Expand the img and masks to simulate doing inference multiple times
                            x_context_local = x_context.unsqueeze(0).repeat(k_local, *([1] * x_context.ndim)).reshape(k_local*batch_size, *x_context.shape[1:])
                            y_context_local = y_context.unsqueeze(0).repeat(k_local, *([1] * y_context.ndim)).reshape(k_local*batch_size, *y_context.shape[1:])
                            x_target_local = x_target.unsqueeze(0).repeat(k_local, *([1] * x_target.ndim)).reshape(k_local*batch_size, *x_target.shape[1:])
                            y_target_local = y_target.unsqueeze(0).repeat(k_local, *([1] * y_target.ndim)).reshape(k_local*batch_size, *y_target.shape[1:])
                            # Prepare input batches
                            p_y_pred = model.forward_test(x_context_local, y_context_local, x_target_local)
                            # Compute log_prob and add it to the list
                            log_prob_flat = p_y_pred.log_prob(y_target_local)
                            log_prob = log_prob_flat.view(k_local, batch_size, *log_prob_flat.shape[1:])
                            log_prob = log_prob.sum(dim=-1)
                            log_prob_parts.append(log_prob)
                            l2_parts.append(torch.sqrt((p_y_pred.loc - y_target_local)**2).mean(list(range(1, y_target_local.ndim))))
                        
                        log_prob = torch.cat(log_prob_parts, dim=0)
                        assert K == log_prob.size(0)
                        unobserved_nll_list.append(-(log_prob.logsumexp(dim=0) - np.log(K)).mean().item())
                        unobserved_l2_list.append(torch.mean(torch.cat(l2_parts, dim=0)).item())
        model.train()
        results = {"valid_loss.NLL": np.mean(nll_list),
                   "valid_loss.L2": np.mean(l2_list),
                   "valid_loss.NLL_std": np.std(nll_list),
                   "valid_loss.L2_std": np.std(l2_list)}
        results.update({"valid_loss.NLL_unobserved": np.mean(unobserved_nll_list),
                        "valid_loss.L2_unobserved": np.mean(unobserved_l2_list),
                        "valid_loss.NLL_unobserved_std": np.std(unobserved_nll_list),
                        "valid_loss.L2_unobserved_std": np.std(unobserved_l2_list)})
        return results