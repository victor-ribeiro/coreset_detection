def _update_grads_val(self, grads_curr=None, first_init=False):
    # update c/ grad médio
    """
    Update the gradient values
    Parameters
    ----------
    grad_currX: OrderedDict, optional
        Gradients of the current element (default: None)
    first_init: bool, optional
        Gradient initialization (default: False)
    perClass: bool
        if True, the function computes the validation gradients using perclass dataloaders
    perBatch: bool
        if True, the function computes the validation gradients of each mini-batch
    """
    self.model.zero_grad()
    embDim = self.model.get_embedding_dim()

    if self.selection_type == "PerClass":
        valloader = self.pcvalloader
    else:
        valloader = self.valloader

    elif grads_curr is not None:
        out_vec = self.init_out - (
            self.eta
            * grads_curr[0][0 : self.num_classes]
            .view(1, -1)
            .expand(self.init_out.shape[0], -1)
        )

        if self.linear_layer:
            out_vec = out_vec - (
                self.eta
                * torch.matmul(
                    self.init_l1,
                    grads_curr[0][self.num_classes :]
                    .view(self.num_classes, -1)
                    .transpose(0, 1),
                )
            )

        loss = self.loss(out_vec, self.y_val.view(-1)).sum()
        l0_grads = torch.autograd.grad(loss, out_vec)[0]
        if self.linear_layer:
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        if self.selection_type == "PerBatch":
            b = int(self.y_val.shape[0] / self.valloader.batch_size)
            l0_grads = torch.chunk(l0_grads, b, dim=0)
            new_t = []
            for i in range(len(l0_grads)):
                new_t.append(torch.mean(l0_grads[i], dim=0).view(1, -1))
            l0_grads = torch.cat(new_t, dim=0)
            if self.linear_layer:
                l1_grads = torch.chunk(l1_grads, b, dim=0)
                new_t = []
                for i in range(len(l1_grads)):
                    new_t.append(torch.mean(l1_grads[i], dim=0).view(1, -1))
                l1_grads = torch.cat(new_t, dim=0)
    torch.cuda.empty_cache()
    if self.linear_layer:
        self.grads_val_curr = torch.mean(
            torch.cat((l0_grads, l1_grads), dim=1), dim=0
        ).view(-1, 1)
    else:
        self.grads_val_curr = torch.mean(l0_grads, dim=0).view(-1, 1)


def eval_taylor_modular(self, grads):
    """
    Evaluate gradients

    Parameters
    ----------
    grads: Tensor
        Gradients

    Returns
    ----------
    gains: Tensor
        Matrix product of two tensors
    """
    grads_val = self.grads_val_curr  # média dos gradientes por batch
    with torch.no_grad():
        gains = torch.matmul(grads, grads_val)
    return gains


def _update_gradients_subset(self, grads, element):
    """
    Update gradients of set X + element (basically adding element to X)
    Note that it modifies the input vector! Also grads is a list! grad_e is a tuple!

    Parameters
    ----------
    grads: list
        Gradients
    element: int
        Element that need to be added to the gradients
    """
    # if isinstance(element, list):
    grads += self.grads_per_elem[element].sum(dim=0)


def greedy_algo(self, K):
    sset = []
    N = self.grads_per_elem.shape[0]
    remainSet = list(range(N))
    t_ng_start = time.time()  # naive greedy start time
    numSelected = 0
    while len(sset) < K:
        # Try Using a List comprehension here!
        rem_grads = self.grads_per_elem[remainSet]
        gains = self.eval_taylor_modular(rem_grads)
        # Update the greedy set and remaining set
        _, indices = torch.sort(gains.view(-1), descending=True)
        bestId = [remainSet[indices[0].item()]]
        greedySet.append(bestId[0])
        remainSet.remove(bestId[0])
        numSelected += 1
        # Update info in grads_currX using element=bestId
        if numSelected == 1:
            grads_curr = self.grads_per_elem[bestId[0]].view(1, -1)
        else:  # If 1st selection, then just set it to bestId grads
            self._update_gradients_subset(grads_curr, bestId)
        # Update the grads_val_current using current greedySet grads
        self._update_grads_val(grads_curr)
    self.logger.debug("Naive Greedy GLISTER total time: %.4f", time.time() - t_ng_start)
    return list(greedySet), [1] * K


def glister():
    # https://github.com/decile-team/cords/blob/main/cords/selectionstrategies/SL/glisterstrategy.py
    # -> compute gradients
    #####################
    # update gradients
    #####################
    # -> greedy algo
    # -> join subsets
    pass
