%NEIGHBORHOOD_FIGS  Six-panel validation of the neighbourhood partition
%   across three different dynamics.
%
%   Columns: three systems, all diagonalizable and nonsingular:
%     1  A = [1 -0.5; 0.4 1]    complex pair 1 +/- 0.45i (worked example)
%     2  A = [0.8 0.3; 0.1 0.6] real eigenvalues 0.9, 0.5 (grid seeds)
%     3  rotation by pi/5       pure rotation, |mu| = 1 (no contraction)
%   Top row (k = 0): partition + seeds + random states coloured by cell.
%   Bottom row (k = 1): everything propagated by x+ = A x + B u0 with
%   u0 = [1;0]; every state keeps its colour (index preservation).
%
%   Writes nb_A.eps ... nb_F.eps (panel letters, row-major) into the
%   Science Robotics submission directory, and APPENDS the tabulated data
%   behind each panel (seeds, sampled states, cell indices) to data_S1.xlsx
%   in the same directory. Run sr_error_figs.m first: it creates the
%   workbook and its README sheet.

clear; close all;
outdir = fullfile('..','Science Robotics submission');
xlsx   = fullfile(outdir, 'data_S1.xlsx');
u0 = [1; 0];                                   % B = I for all systems
sysrows = cell(3, 9);                          % Fig3_systems sheet

S7 = [-1 -2 -3  2 -1  2  3;                    % the seven seeds of I_A
      -2 -3 -1  1  2  3  1];

sys(1).A = [1 -0.5; 0.4 1];   sys(1).seeds = S7;
sys(1).tag = '$\mu_{1,2}=1\pm0.45\mathrm{i}$';
sys(2).A = [0.8 0.3; 0.1 0.6];                 % seeds built below (grid)
sys(2).tag = '$\mu_1=0.9,\ \mu_2=0.5$';
th = pi/5;
sys(3).A = [cos(th) -sin(th); sin(th) cos(th)]; sys(3).seeds = S7;
sys(3).tag = '$\mu_{1,2}=e^{\pm\mathrm{i}\pi/5}$';

% Okabe-Ito colourblind-safe palette (7 hues)
col = [  0 114 178; 230 159   0;   0 158 115; 204 121 167;
        86 180 233; 213  94   0; 240 228  66] / 255;
mk  = {'o','s','d','^','v','>','p'};
letters = 'ABCDEF';
fs = 11;
boxA = polyshape([-5 -5; 5 -5; 5 5; -5 5]);

rng(1);
for s = 1:3
    A = sys(s).A;
    [Pi, blk, isreal2] = eigdirs(A);
    if s == 2                                  % grid seeds in eigen-coords
        g1 = [-2.5 0 2.5];  g2 = [-2 2];
        [Z1, Z2] = ndgrid(g1, g2);
        V = Pi \ [Z1(:).'; Z2(:).'];
        assert(all(abs(V(:)) <= 5), 'grid seeds fall outside the box');
    else
        V = sys(s).seeds;
    end
    J = size(V,2);

    N   = 220;
    X   = 10*rand(2,N) - 5;
    cls  = classify(Pi, blk, V, X);
    Vp   = A*V + u0;
    Xp   = A*X + u0;
    clsP = classify(Pi, blk, Vp, Xp);
    assert(isequal(cls, clsP), 'index preservation violated (system %d)', s);

    % ---- data S1: seeds and sampled states behind this column of Figure 3
    pan = sprintf('Fig3%c%c', letters(s), letters(s+3));
    writetable(table((1:J)', V(1,:)', V(2,:)', Vp(1,:)', Vp(2,:)', ...
        'VariableNames', {'seed_id','x1_k0','x2_k0','x1_k1','x2_k1'}), ...
        xlsx, 'Sheet', [pan '_seeds']);
    writetable(table((1:N)', X(1,:)', X(2,:)', cls(:), Xp(1,:)', Xp(2,:)', clsP(:), ...
        'VariableNames', {'state_id','x1_k0','x2_k0','cell_k0','x1_k1','x2_k1','cell_k1'}), ...
        xlsx, 'Sheet', [pan '_states']);
    lam = eig(A);
    sysrows(s,:) = {s, pan, A(1,1), A(1,2), A(2,1), A(2,2), ...
                    real(lam(1)), imag(lam(1)), J};

    corn = (A*[-5 5 5 -5; -5 -5 5 5] + u0)';
    boxB = polyshape(corn);
    lim  = max(abs(corn(:))) + 0.4;

    % ---- top panel: k = 0
    f = figure('Units','centimeters','Position',[2 2 8.6 8.2]);
    draw_partition(Pi, blk, V, boxA, col, isreal2);
    draw_states(X, cls, col, J);
    draw_seeds(V, col, mk);
    text(-4.7, 4.4, ['$k=0$;\ ' sys(s).tag], 'Interpreter','latex', ...
         'FontSize',fs-1);
    finish_axes(fs, [-5.2 5.2], [-5.2 5.2]);
    print(f, fullfile(outdir, ['nb_' letters(s)]), '-depsc2');

    % ---- bottom panel: k = 1
    f = figure('Units','centimeters','Position',[2 2 8.6 8.2]);
    draw_partition(Pi, blk, Vp, boxB, col, isreal2);
    draw_states(Xp, cls, col, J);                 % colours from k = 0
    draw_seeds(Vp, col, mk);
    text(-lim+0.35, lim-0.75, '$k=1$;\ $u_0=[1,\,0]^{\top}$', ...
         'Interpreter','latex', 'FontSize',fs-1);
    finish_axes(fs, [-lim lim], [-lim lim]);
    print(f, fullfile(outdir, ['nb_' letters(s+3)]), '-depsc2');
end
writetable(cell2table(sysrows, 'VariableNames', {'system','panels','A11','A12','A21','A22', ...
    'eigenvalue_real','eigenvalue_imag','num_seeds'}), xlsx, 'Sheet', 'Fig3_systems');
fprintf('nb_A.eps ... nb_F.eps written to %s\n', outdir);
fprintf('data S1 sheets for Figure 3 appended to %s\n', xlsx);

%% ------------------------------------------------------------------ helpers
function [Pi, blk, isreal2] = eigdirs(A)
% Rows of Pi are Re/Im of a normalized complex left eigenvector (one block,
% d = 2) or the two normalized real left eigenvectors (two blocks, d = 1).
[W, D] = eig(A.');
lam = diag(D);
if isreal(lam)
    w1 = W(:,1)/norm(W(:,1));  w2 = W(:,2)/norm(W(:,2));
    Pi = [w1.'; w2.'];
    blk = {1, 2};  isreal2 = true;
else
    w = W(:,1)/norm(W(:,1));                   % w.' A = lam w.'
    Pi = [real(w.'); imag(w.')];
    blk = {1:2};  isreal2 = false;
end
end

function cls = classify(Pi, blk, V, X)
% Per-block argmin of the eigen-direction distances; the common winner is
% unique for eigen-separable seed sets (smallest index on ties).
J = size(V,2);  N = size(X,2);
common = true(J, N);
for b = 1:numel(blk)
    P = Pi(blk{b},:);
    D = zeros(J, N);
    for i = 1:J
        D(i,:) = vecnorm(P*(X - V(:,i)), 2, 1);   % column-wise, also for 1-row P
    end
    common = common & (D <= min(D,[],1) + 1e-9);
end
assert(all(any(common,1)), 'a sample has no common per-block minimiser');
[~, cls] = max(common, [], 1);                 % first true = smallest index
end

function draw_partition(Pi, ~, V, box, col, isreal2)
hold on;
if ~isreal2
    % single 2-D block: Euclidean Voronoi in z = Pi*x coordinates
    zs = (Pi*V)';
    th = (0:7)'*pi/4;
    [vv, cc] = voronoin([zs; 300*[cos(th) sin(th)]]);
    for i = 1:size(V,2)
        vidx = cc{i};
        if any(vidx == 1), continue; end
        px = (Pi \ vv(vidx,:)')';
        plot_cell(px, box, col(i,:));
    end
else
    % two 1-D blocks: cells are rectangles in z-space between midpoints
    zs = Pi*V;
    for i = 1:size(V,2)
        px = (Pi \ cell_rect(zs, i)')';
        plot_cell(px, box, col(i,:));
    end
end
plot(box, 'FaceColor','none', 'EdgeColor','k', 'LineWidth', 0.8);
end

function R = cell_rect(zs, i)
% Axis-aligned z-space rectangle of seed i, bounded by midpoints to the
% neighbouring distinct coordinate values (outer bound +/- 300).
R = zeros(4,2);
for a = 1:2
    vals = uniquetol(zs(a,:), 1e-9);
    zi = zs(a,i);
    lo = vals(vals < zi - 1e-9);  hi = vals(vals > zi + 1e-9);
    if isempty(lo), lb = -300; else, lb = (max(lo)+zi)/2; end
    if isempty(hi), ub =  300; else, ub = (min(hi)+zi)/2; end
    if a == 1, R([1 4],1) = lb;  R([2 3],1) = ub;
    else,      R([1 2],2) = lb;  R([3 4],2) = ub;
    end
end
end

function plot_cell(px, box, c)
ctr = mean(px,1);
[~,o] = sort(atan2(px(:,2)-ctr(2), px(:,1)-ctr(1)));
p = intersect(polyshape(px(o,1), px(o,2)), box);
if p.NumRegions > 0
    plot(p, 'FaceColor', c, 'FaceAlpha', 0.30, ...
         'EdgeColor', [0.35 0.35 0.35], 'LineWidth', 0.6);
end
end

function draw_states(X, cls, col, J)
for i = 1:J
    m = (cls == i);
    scatter(X(1,m), X(2,m), 10, 0.8*col(i,:), 'filled', ...
            'MarkerEdgeColor','none');
end
end

function draw_seeds(V, col, mk)
for i = 1:size(V,2)
    plot(V(1,i), V(2,i), mk{i}, 'MarkerFaceColor', col(i,:), ...
         'MarkerEdgeColor','k', 'MarkerSize', 8, 'LineWidth', 0.7);
end
end

function finish_axes(fs, xl, yl)
axis equal; xlim(xl); ylim(yl);
xlabel('$x_1$','Interpreter','latex','FontSize',fs);
ylabel('$x_2$','Interpreter','latex','FontSize',fs);
set(gca,'FontSize',fs-1,'Layer','top');
end
