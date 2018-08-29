import itertools

import numpy as np

import utils


class TensorProjector(object):
    def __init__(self, base, rank):
        self.base = base
        self.rank = rank
        self.all = slice(0, base, 1)

    def tensorize(self, v):
        #TODO: Use values.reshape for panda series
        return v.reshape(*[self.base] * self.rank)

    def project_vector(self, v, const_indices):
        v_tensor = self.tensorize(v)
        return self.project_tensor(v_tensor, const_indices)

    def project_tensor(self, v, const_indices):
        projection_desc = [self.all] * self.rank
        for index, value in const_indices:
            projection_desc[index] = value
        return v[projection_desc]


def generate_all_standard_projections(w, base, full_rank, proj_rank, tagged=True):
    assert len(w) == base**full_rank
    assert 0 < proj_rank <= full_rank

    res_rank = full_rank - proj_rank
    fixed_setup = (0,) * res_rank # standard projections: absent bystanders
    idx_options = tuple(itertools.combinations(range(full_rank), r=res_rank))
    contexts = tuple(tuple(zip(io, fixed_setup)) for io in idx_options)

    tp = TensorProjector(base, full_rank)
    wt = tp.tensorize(w)

    projections = np.array(tuple(
        tp.project_tensor(wt, ctx).flatten()
        for ctx in contexts
    ))
    if not tagged:
        return projections

    tags = tuple(utils.format_context(ctx, full_rank) for ctx in contexts)
    #print("Projections:", projections)
    #   print("Tags: ", tags)
    return projections, tags


def generate_all_projections(w, base, full_rank, proj_rank, tagged=True):
    assert base > 0
    assert len(w) == base**full_rank
    assert 0 < proj_rank <= full_rank

    res_rank = full_rank - proj_rank
    # (Non-)standard projections: all possible bystander backgrounds
    fixed_setups = tuple(itertools.product(range(base), repeat=res_rank))
    idx_options = tuple(itertools.combinations(range(full_rank), r=res_rank))
    contexts = tuple(
        tuple(zip(io, fixed_setup))
        for io, fixed_setup in itertools.product(idx_options, fixed_setups)
    )

    tp = TensorProjector(base, full_rank)
    wt = tp.tensorize(w)

    projections = np.array(tuple(
        tp.project_tensor(wt, ctx).flatten()
        for ctx in contexts
    ))
    if not tagged:
        return projections

    tags = tuple(utils.format_context(ctx, full_rank) for ctx in contexts)
    return projections, tags


def main():
    print('Assume 5 different species, either present or absent (l = 2).')
    print('Construct the fitness vector w_flat, indexed by {b_0 b_1 ... b_4}')
    l, rank = 2, 5
    w_flat = np.arange(0, l**rank)
    print(w_flat)
    
    print('\nTransform w_flat into 5-D tensor with l = 2 elements per dimension.')
    w_tensor = w_flat.reshape(*[l] * rank)
    print(w_tensor)

    print('\nSelect a 2x2 slice of w where (b_0, b_2, b_3) = (0, 1, 1)')
    w_proj = w_tensor[0, :, 1, 1, :]
    print(w_proj)

    print('\nSame procedure but using TensorProjector')
    const_dims = [(0, 0), (2, 1), (3, 1)]
    cp = TensorProjector(2, 5)
    w_tensor2 = cp.project_vector(w_flat, const_dims)
    print(w_tensor2)
    
    print('\nGenerate all standard projections of size 3)')
    projections, tags = generate_all_standard_projections(w_flat, l, rank, 3)
    for p, t in zip(projections, tags):
        print(t, p)

    print('\nGenerate all projections of size 3)')
    projections, tags = generate_all_projections(w_flat, l, rank, 3)
    for p, t in zip(projections, tags):
        print(t, p)


if __name__ == '__main__':
    main()
