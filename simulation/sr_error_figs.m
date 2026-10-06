%SR_ERROR_FIGS  Improved error-bound figures for the Science Robotics manuscript.
%
%   Recomputes E1-E3 of bisim_run_all.m with the IDENTICAL random stream
%   (rng(0), same call order, E1 included only to advance the stream) so the
%   plotted data match the numbers quoted in the text, then redraws
%   bisim_partition.eps / bisim_truncation.eps with:
%     - one colour per sweep (bound solid, measured dashed);
%     - explicit fixed-parameter values in the legend;
%     - the max-over-trajectories statistic named on the y-axis;
%     - slope / decay-rate annotations.
%   Writes the EPS files into the Science Robotics submission directory only;
%   the IEEE working-copy figures are left untouched.

clear; close all; rng(0);
outdir = fullfile('..','Science Robotics submission');
S = bisim_setup();

%% E1 -- run only to keep the random stream identical to bisim_run_all.m
eps0 = 0.25;  G = bisim_grid(S, eps0);
Ntraj = 20000;  Kh = 25;  ok = 0;  tot = 0;
for t = 1:Ntraj
    x = (S.Ibox(2)-S.Ibox(1))*rand(3,1) + S.Ibox(1);
    i0 = bisim_project(S, G, x, 0, zeros(3,1));
    wxi = zeros(3,1);
    for k = 1:Kh
        u = S.U(:, randi(size(S.U,2)));
        x   = S.A*x + S.B*u;
        wxi = S.A*wxi + S.B*u;
        ik  = bisim_project(S, G, x, k, wxi);
        tot = tot + 1;
        ok  = ok + isequal(ik, i0);
    end
end
fprintf('E1 index preserved: %.4f%% (%d/%d)\n', 100*ok/tot, ok, tot);

%% E2 -- partition error vs covering radius
epsv = 0.6*0.5.^(0:5);
emp2 = zeros(size(epsv));  bnd2 = zeros(size(epsv));
for e = 1:numel(epsv)
    Ge = bisim_grid(S, epsv(e));
    bnd2(e) = S.M*sqrt(S.r)*epsv(e)/S.alpha;
    worst = 0;
    for t = 1:4000
        x0 = (S.Ibox(2)-S.Ibox(1))*rand(3,1) + S.Ibox(1);
        i0 = bisim_project(S, Ge, x0, 0, zeros(3,1));
        v0 = Ge.IA(:, blk2lin(Ge, i0));
        x = x0;  v = v0;
        for k = 0:20
            worst = max(worst, norm(x - v));
            u = S.U(:, randi(size(S.U,2)));
            x = S.A*x + S.B*u;   v = S.A*v + S.B*u;
        end
    end
    emp2(e) = worst;
    fprintf('E2 eps=%6.4f  measured=%7.4f  bound=%7.4f  ratio=%.3f\n', ...
        epsv(e), emp2(e), bnd2(e), emp2(e)/bnd2(e));
end

%% E3 -- truncation error vs horizons
K1f = 30;  K2v = 0:10;  K1v = 0:10;  K2f = 30;
G3  = bisim_grid(S, eps0);  rI = G3.rI;
emp3b = zeros(size(K2v));  th3b = zeros(size(K2v));
emp3a = zeros(size(K1v));  th3a = zeros(size(K1v));
for a = 1:numel(K2v)
    th3b(a) = S.M*rI*S.rho^(K1f+1) + S.M*S.rU*S.rho^K2v(a)/(1-S.rho);
    emp3b(a) = trunc_emp(S, G3, K1f, K2v(a), 4000, 45);
end
for a = 1:numel(K1v)
    th3a(a) = S.M*rI*S.rho^(K1v(a)+1) + S.M*S.rU*S.rho^K2f/(1-S.rho);
    emp3a(a) = trunc_emp(S, G3, K1v(a), K2f, 4000, 45);
end

%% E4 -- precision / complexity trade-off (deterministic, no randomness)
nU = size(S.U,2);
E4 = [];
for e = 1:numel(epsv)
    Ge = bisim_grid(S, epsv(e));
    for K1 = [2 4 6 8]
        for K2 = 1:6
            tau  = S.M*Ge.rI*S.rho^(K1+1) + S.M*S.rU*S.rho^K2/(1-S.rho);
            taup = S.M*sqrt(S.r)*epsv(e)/S.alpha + tau;
            nS   = Ge.J*(K1+1)*nU^K2;
            E4(end+1,:) = [nS, taup, epsv(e), K1, K2]; %#ok<SAGROW>
        end
    end
end
[~,o] = sort(E4(:,1));  E4 = E4(o,:);
par = true(size(E4,1),1);                       % Pareto front
best = inf;
for q = 1:size(E4,1)
    if E4(q,2) < best, best = E4(q,2); else, par(q) = false; end
end
tgts = [0.5 1 2 4 8];                           % cheapest certified tau' <= tgt
tgt  = tgts(find(tgts >= min(E4(:,2)), 1));
sel  = E4(E4(:,2) <= tgt, :);
[~,b] = min(sel(:,1));  star = sel(b,:);  star_tgt = tgt;
fprintf('E4: %d configurations, %d on the Pareto front\n', size(E4,1), sum(par));
fprintf('E4: tau''<=%g cheapest: %d states (eps=%.4f, K1=%d, K2=%d, tau''=%.3f)\n', ...
    star_tgt, star(1), star(3), star(4), star(5), star(2));

%% E5 -- conditioning sweep. The first three shears reproduce the random
%% stream (and hence the numbers quoted in the text) of bisim_run_all.m;
%% the remaining shears are appended afterwards to densify the curve.
shears = [0.5 1.5 3.0, 0.75 1.0 2.0 2.5];
Mv = zeros(size(shears)); taupv = Mv; wv = Mv;
for q = 1:numel(shears)
    sh = shears(q);
    Ss = bisim_setup(sh);  Gs = bisim_grid(Ss, eps0);
    taupv(q) = Ss.M*sqrt(Ss.r)*eps0/Ss.alpha + ...
               Ss.M*Gs.rI*Ss.rho^(6+1) + Ss.M*Ss.rU*Ss.rho^4/(1-Ss.rho);
    worst = 0;
    for t = 1:3000
        x0 = (Ss.Ibox(2)-Ss.Ibox(1))*rand(3,1) + Ss.Ibox(1);
        i0 = bisim_project(Ss, Gs, x0, 0, zeros(3,1));
        v = Gs.IA(:, blk2lin(Gs, i0));  x = x0;
        for k = 0:20
            worst = max(worst, norm(x-v));
            u = Ss.U(:, randi(size(Ss.U,2)));
            x = Ss.A*x + Ss.B*u;  v = Ss.A*v + Ss.B*u;
        end
    end
    Mv(q) = Ss.M;  wv(q) = worst;
    fprintf('E5 shear=%.2f  M=%6.3f  tau''=%7.3f  measured=%6.3f\n', ...
        sh, Ss.M, taupv(q), worst);
end
[Mv, o] = sort(Mv);  taupv = taupv(o);  wv = wv(o);  shears = shears(o);

%% export -- tabulated data underlying Figure 4 and the numbers quoted in the
%% text, written as data S1 (one sheet per figure panel, as Science Robotics
%% requires). This script CREATES the workbook; run neighborhood_figs.m
%% afterwards to append the Figure 3 sheets to the same file.
xlsx = fullfile(outdir, 'data_S1.xlsx');
if exist(xlsx, 'file'), delete(xlsx); end

readme = { ...
    'README',             '-',        'Tabulated data underlying the figures of the main text. One sheet per figure panel; column headers name each quantity. Random seeds: rng(0) for Figure 4 (sr_error_figs.m), rng(1) for Figure 3 (neighborhood_figs.m).';
    'System_parameters',  'Fig. 4',   'Three-dimensional example of Figure 4: spectral radius, power-bound constant M, sigma_min(Pi), cond(Pi), number of eigen-directions, input-set size, covering radius and seed-set radius used in the sweeps.';
    'Text_index_preservation', 'Results text', 'Index-preservation check quoted in the text: number of random trajectories, horizon, number of one-step checks and number preserved.';
    'Fig4A_partition',    'Fig. 4A',  'Partition error versus covering radius epsilon: measured maximum over 4000 random trajectories of 21 steps, and the bound of Lemma 3.';
    'Fig4B_truncation',   'Fig. 4B',  'Truncation error versus horizon: K1 sweep at fixed K2 and K2 sweep at fixed K1, measured maximum over 4000 random input strings of 45 steps, and the precision tau of Theorem 3.';
    'Fig4C_tradeoff',     'Fig. 4C',  'All (epsilon, K1, K2) configurations with the number of symbolic states and the certified precision tau-prime; flags mark the Pareto front and the cheapest configuration certifying tau-prime <= target.';
    'Fig4D_conditioning', 'Fig. 4D',  'Conditioning sweep: shear parameter of the eigenvector basis, resulting M = kappa(V), certificate tau-prime and measured error (maximum over 3000 random trajectories of 21 steps).';
    'Fig3AD_seeds',       'Fig. 3A,D','Seeds of the worked example (complex pair) at k = 0 and after the common input u0 (k = 1).';
    'Fig3AD_states',      'Fig. 3A,D','Random states of the worked example with their cell index at k = 0 and k = 1 (identical by index preservation).';
    'Fig3BE_seeds',       'Fig. 3B,E','Seeds of the system with two distinct real eigenvalues at k = 0 and k = 1.';
    'Fig3BE_states',      'Fig. 3B,E','Random states of the real-eigenvalue system with cell indices at k = 0 and k = 1.';
    'Fig3CF_seeds',       'Fig. 3C,F','Seeds of the pure-rotation system at k = 0 and k = 1.';
    'Fig3CF_states',      'Fig. 3C,F','Random states of the pure-rotation system with cell indices at k = 0 and k = 1.';
    'Fig3_systems',       'Fig. 3',   'System matrices and eigenvalues of the three dynamics of Figure 3.'};
writetable(cell2table(readme, 'VariableNames', {'sheet','figure_panel','description'}), ...
    xlsx, 'Sheet', 'README');

pnames = {'rho_A'; 'M_cond_V'; 'alpha_sigma_min_Pi'; 'cond_Pi'; 'r_eigen_directions'; ...
          'num_inputs'; 'r_U_max_norm_Bu'; 'eps0'; 'r_I_at_eps0'; 'I_box_half_width'};
pvals  = [S.rho; S.M; S.alpha; cond(S.Pi); S.r; nU; S.rU; eps0; G3.rI; S.Ibox(2)];
writetable(table(pnames, pvals, 'VariableNames', {'parameter','value'}), ...
    xlsx, 'Sheet', 'System_parameters');

writetable(table(Ntraj, Kh, tot, ok, 100*ok/tot, 'VariableNames', ...
    {'num_trajectories','horizon_steps','num_checks','num_index_preserved','percent_preserved'}), ...
    xlsx, 'Sheet', 'Text_index_preservation');

writetable(table(epsv(:), emp2(:), bnd2(:), emp2(:)./bnd2(:), 'VariableNames', ...
    {'epsilon','measured_max_partition_error','bound_M_sqrt_r_eps_over_alpha','ratio_measured_over_bound'}), ...
    xlsx, 'Sheet', 'Fig4A_partition');

writetable(table([repmat("K1",numel(K1v),1); repmat("K2",numel(K2v),1)], ...
    [K1v(:); K2v(:)], [repmat(K2f,numel(K1v),1); repmat(K1f,numel(K2v),1)], ...
    [th3a(:); th3b(:)], [emp3a(:); emp3b(:)], 'VariableNames', ...
    {'swept_parameter','swept_value','fixed_other_parameter','tau_bound','measured_max_truncation_error'}), ...
    xlsx, 'Sheet', 'Fig4B_truncation');

writetable(table(E4(:,1), E4(:,2), E4(:,3), E4(:,4), E4(:,5), par, ...
    ismember(E4, star, 'rows'), repmat(star_tgt, size(E4,1), 1), 'VariableNames', ...
    {'num_symbolic_states','tau_prime','epsilon','K1','K2','on_pareto_front','cheapest_certified','certification_target'}), ...
    xlsx, 'Sheet', 'Fig4C_tradeoff');

writetable(table(shears(:), Mv(:), taupv(:), wv(:), 'VariableNames', ...
    {'shear','M_cond_V','certificate_tau_prime','measured_max_error'}), ...
    xlsx, 'Sheet', 'Fig4D_conditioning');
fprintf('data S1 sheets for Figure 4 written to %s\n', xlsx);

%% figures
fs = 10;
blue = [0 114 178]/255;                        % Okabe-Ito, matches Figure 3
verm = [213  94   0]/255;

f1 = figure('Units','centimeters','Position',[2 2 8.6 6.6]);
loglog(epsv, bnd2, '-o','Color',blue,'LineWidth',1.2,'MarkerSize',4.5, ...
       'MarkerFaceColor',blue); hold on; grid on;
loglog(epsv, emp2, '--s','Color',verm,'LineWidth',1.2,'MarkerSize',4.5, ...
       'MarkerFaceColor','w');
text(0.09, 0.62, 'slope $1$', 'Interpreter','latex', 'FontSize',fs-1, ...
     'Rotation', 33);
xlabel('covering radius $\varepsilon$','Interpreter','latex','FontSize',fs);
ylabel('$\max_k\|x_k-\hat{h}(P_\xi(x_k))\|$','Interpreter','latex','FontSize',fs);
legend({'bound $M\sqrt{r}\,\varepsilon/\underline{\alpha}$', 'measured'}, ...
       'Interpreter','latex','Location','northwest','FontSize',fs-1);
set(gca,'FontSize',fs-1);
print(f1, fullfile(outdir,'bisim_partition'), '-depsc2');

f2 = figure('Units','centimeters','Position',[2 2 8.6 6.6]);
semilogy(K1v, th3a, '-o','Color',verm,'LineWidth',1.2,'MarkerSize',4.5, ...
         'MarkerFaceColor',verm); hold on; grid on;
semilogy(K1v, emp3a,'--s','Color',verm,'LineWidth',1.2,'MarkerSize',4.5, ...
         'MarkerFaceColor','w');
semilogy(K2v, th3b, '-^','Color',blue,'LineWidth',1.2,'MarkerSize',4.5, ...
         'MarkerFaceColor',blue);
semilogy(K2v, emp3b,'--v','Color',blue,'LineWidth',1.2,'MarkerSize',4.5, ...
         'MarkerFaceColor','w');
text(0.3, 0.04, 'decay rate $\rho(A)=0.60$', 'Interpreter','latex', ...
     'FontSize',fs-1, 'HorizontalAlignment','left');
xlabel('$K_1$ or $K_2$','Interpreter','latex','FontSize',fs);
ylabel('$\max\|\tilde{h}-\hat{h}\|$','Interpreter','latex','FontSize',fs);
legend({sprintf('$\\tau$, $K_1$ sweep ($K_2{=}%d$)', K2f), ...
        'measured, $K_1$ sweep', ...
        sprintf('$\\tau$, $K_2$ sweep ($K_1{=}%d$)', K1f), ...
        'measured, $K_2$ sweep'}, ...
       'Interpreter','latex','Location','northeast','FontSize',fs-2);
set(gca,'FontSize',fs-1);
print(f2, fullfile(outdir,'bisim_truncation'), '-depsc2');

f3 = figure('Units','centimeters','Position',[2 2 8.6 6.6]);
loglog(E4(:,1), E4(:,2), '.', 'Color',[.65 .65 .65],'MarkerSize',8); hold on; grid on;
loglog(E4(par,1), E4(par,2), '-o','Color',blue,'LineWidth',1.2, ...
       'MarkerSize',4.5,'MarkerFaceColor',blue);
loglog(star(1), star(2), 'p', 'Color',verm, 'MarkerFaceColor',verm, ...
       'MarkerSize',13);
text(star(1), star(2)*2.6, ...
     sprintf('$\\tau''\\le %g$: $%.1f\\times10^{%d}$ states', star_tgt, ...
             star(1)/10^floor(log10(star(1))), floor(log10(star(1)))), ...
     'Interpreter','latex','FontSize',fs-1,'HorizontalAlignment','center');
xlabel('number of symbolic states $J(K_1+1)|U|^{K_2}$','Interpreter','latex','FontSize',fs);
ylabel('precision $\tau''$','Interpreter','latex','FontSize',fs);
legend({'all configurations','Pareto front', ...
        sprintf('cheapest $\\tau''\\le %g$', star_tgt)}, ...
       'Interpreter','latex','Location','southwest','FontSize',fs-1);
set(gca,'FontSize',fs-1);
print(f3, fullfile(outdir,'bisim_tradeoff'), '-depsc2');

f4 = figure('Units','centimeters','Position',[2 2 8.6 6.6]);
loglog(Mv, taupv, '-o','Color',blue,'LineWidth',1.2,'MarkerSize',4.5, ...
       'MarkerFaceColor',blue); hold on; grid on;
loglog(Mv, wv, '--s','Color',verm,'LineWidth',1.2,'MarkerSize',4.5, ...
       'MarkerFaceColor','w');
xlabel('eigenvector conditioning $M=\kappa(V)$','Interpreter','latex','FontSize',fs);
ylabel('error','Interpreter','latex','FontSize',fs);
legend({'certificate $\tau''$','measured error'}, ...
       'Interpreter','latex','Location','northwest','FontSize',fs-1);
set(gca,'FontSize',fs-1);
print(f4, fullfile(outdir,'bisim_conditioning'), '-depsc2');

fprintf('bisim_partition/truncation/tradeoff/conditioning .eps written to %s\n', outdir);

%% ------------------------------------------------------------------ helpers
function lin = blk2lin(G, idx)
sz = []; sub = [];
for i = 1:numel(G.blk)
    for q = 1:numel(G.blk{i})
        sz(end+1) = numel(G.blk{i}{q}); %#ok<AGROW>
        sub(end+1) = idx{i}(q);         %#ok<AGROW>
    end
end
c = num2cell(sub);
lin = sub2ind(sz, c{:});
end

function w = trunc_emp(S, G, K1, K2, Ntr, Kh)
% empirical max over random input strings of ||htilde - hhat||
w = 0;
for t = 1:Ntr
    j  = randi(G.J);   v0 = G.IA(:,j);
    us = S.U(:, randi(size(S.U,2), 1, Kh));
    acc_full = zeros(3,1);
    hist = zeros(3,0);
    for k = 1:Kh
        u = us(:,k);
        acc_full = S.A*acc_full + S.B*u;
        hist = [hist, S.B*u]; %#ok<AGROW>
        m = min(K2, size(hist,2));
        acc_K2 = zeros(3,1);
        for i = 0:m-1
            acc_K2 = acc_K2 + S.A^i*hist(:,end-i);
        end
        vk_true = S.A^k*v0;
        if k <= K1, vk_tr = vk_true; else, vk_tr = zeros(3,1); end
        w = max(w, norm((vk_tr + acc_K2) - (vk_true + acc_full)));
    end
end
end
