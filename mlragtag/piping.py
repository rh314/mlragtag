import numpy as np
import torch


class PipeOp():
    def build(self):
        pass


class Linear(PipeOp):
    def __init__(self):
        self.settings = {}
        self.required = ('in', 'out')
        self.optional = ()

    def build(self):
        settings = self.settings
        in_f, out_f = settings['in'], settings['out']
        self.linear = torch.nn.Linear(in_f, out_f)

    def apply(self, inpt):
        return self.linear(inpt)


class LReLu(PipeOp):
    def __init__(self):
        self.settings = {'slope': 0.3}
        self.required = ()
        self.optional = ('slope')

    def apply(self, inpt):
        slope = self.settings['slope']
        return torch.nn.functional.leaky_relu(inpt, slope)


class Function(PipeOp):
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.can_step = True

    def apply(self, inpt):
        return self.func(inpt)

    def apply_step(self, inpt, stack, first_step):
        if first_step:
            stack.append([self, inpt, 0])
        else:
            if stack[-1][2] == 0:
                inpt = stack[-1][1]
                res = self.func(inpt)
                stack[-1][2] = 1
                stack[-1][1] = res
            else:
                stack[-2][1] = stack[-1][1]
                stack.pop()


class Chain(PipeOp):
    def __init__(self, ops):  # args must be a tuple
        for op in ops:
            assert isinstance(op, PipeOp)
        self.ops = ops
        self.can_step = True

    def apply(self, inpt):
        res = inpt
        for op in self.ops:
            res = op.apply(res)
        return res

    def apply_step(self, inpt, stack, first_step):
        if first_step:
            stack.append([self, inpt, -1])
        else:
            ops = self.ops
            stack[-1][2] += 1
            i = stack[-1][2]
            if i < len(ops):
                inpt = stack[-1][1]
                op = ops[i]
                if op.can_step:
                    op.apply_step(inpt, stack, True)
                else:
                    res = op.apply(inpt)
                    stack[-1][1] = res
            else:
                stack[-2][1] = stack[-1][1]
                stack.pop()


class PCDropout(PipeOp):
    def __init__(self):
        self.settings = {'r': 0.5}
        self.required = ()
        self.optional = 'r'

    def apply(self, inpt):
        r = self.settings['r']
        dim_r, dim_c = inpt.shape
        if np.random.rand() < r:
            mask = torch.ones(dim_r, dim_c)
            # Want want dim_c + 1 instead of dim_c here,
            # to sensibly handle the case where `r` close to 1.
            idx = np.random.randint(1, dim_c + 1)
            mask[:, idx:] = 0
            return inpt * mask
        else:
            return inpt
