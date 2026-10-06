function idx = bisim_project(S, G, x, k, wxi)
%BISIM_PROJECT  Multi-directional projection P_xi of Definition 3.
%
%   Returns the per-block grid indices of the representative point that
%   minimises d_i(x, hhat(v')) simultaneously for every eigen-direction.
%
%   At layer k the observations of the layer are Lam_i^k * G_i + pi_i(w_xi),
%   so the minimiser is found by mapping x back through Lam_i^k and rounding
%   on the original grid, which is exact because Lam_i is a similarity.

y = S.Pi*(x - wxi);
idx = cell(1,S.r);
row = 1;
for i = 1:S.r
    d  = S.dim(i);
    yi = y(row:row+d-1);
    zi = S.Lam{i}^k \ yi;                    % undo the layer transformation
    ax = G.blk{i};
    id = zeros(1,d);
    for q = 1:d
        [~, id(q)] = min(abs(ax{q} - zi(q)));   % ties -> smallest index
    end
    idx{i} = id;
    row = row + d;
end
end
