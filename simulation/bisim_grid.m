function G = bisim_grid(S, eps_cov)
%BISIM_GRID  Eigen-separable representative set of Lemma 2.
%
%   Encloses Pi(I) in a box, grids each block B_i with spacing
%   2*eps/sqrt(d_i) so that every point of B_i is within Euclidean
%   distance eps of a grid point, and returns I_A = Pi^{-1}(G_1 x ... x G_r).

% bounding box of Pi(I) over the vertices of the cube I
[c1,c2,c3] = ndgrid(S.Ibox, S.Ibox, S.Ibox);
Vtx  = [c1(:) c2(:) c3(:)]';
Ycor = S.Pi*Vtx;
lo   = min(Ycor,[],2);  hi = max(Ycor,[],2);

G.blk = cell(1,S.r);  G.lo = cell(1,S.r);  G.h = zeros(1,S.r);
row = 1;  axes_ = cell(1,0);
for i = 1:S.r
    d  = S.dim(i);
    h  = 2*eps_cov/sqrt(d);                  % spacing in block i
    G.h(i) = h;
    lob = lo(row:row+d-1);  hib = hi(row:row+d-1);
    ax = cell(1,d);
    for q = 1:d
        n = max(1, ceil((hib(q)-lob(q))/h) + 1);
        ax{q} = lob(q) + (0:n-1)*h;
    end
    G.lo{i}  = lob;
    G.blk{i} = ax;
    axes_ = [axes_, ax];
    row = row + d;
end

% Cartesian product -> representative points in the original coordinates
grids = cell(1,3);
[grids{:}] = ndgrid(axes_{:});
Y = zeros(3, numel(grids{1}));
for q = 1:3, Y(q,:) = grids{q}(:)'; end
G.Y  = Y;                                    % eigen-coordinates of I_A
G.IA = S.Pi\Y;                               % I_A itself
G.J  = size(Y,2);
G.rI = max(vecnorm(G.IA));
G.eps = eps_cov;
end
