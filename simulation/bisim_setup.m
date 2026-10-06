function S = bisim_setup(shear)
%BISIM_SETUP  Numerical example for the approximate finite abstraction.
%
%   Builds the 3-D discrete-time linear system with
%     - one real eigenvalue and one complex-conjugate pair, so that the set
%       of eigen-directions has r = 2 members with d_1 = 1 and d_2 = 2;
%     - spectral radius rho < 1, so Assumption 2 holds;
%     - ||A|| > 1, so any bound based on ||A||^k is vacuous and only the
%       M*rho^k bound of (13) applies.
%
%   SHEAR (optional) scales the non-orthogonality of the eigenvector basis
%   and is used to produce the well/ill-conditioned rows of Table I.

if nargin < 1, shear = 1.5; end

% ---- prescribed spectrum -------------------------------------------------
mu_real = 0.6;                       % real eigenvalue
r_c     = 0.5;  th_c = pi/4;         % complex pair 0.5*exp(+-i*pi/4)
S.rho   = max(abs([mu_real, r_c]));  % spectral radius

Dr = blkdiag(mu_real, r_c*[cos(th_c) -sin(th_c); sin(th_c) cos(th_c)]);

% ---- non-normal real basis ----------------------------------------------
% A nilpotent (strictly upper triangular) shear is used so that the
% conditioning of the eigenvector basis grows with SHEAR, which is what
% drives ||A|| above 1 while the spectrum stays inside the unit circle.
Nil = [0 1 0; 0 0 1; 0 0 0];
Wb  = eye(3) + shear*Nil;
S.A = Wb*Dr/Wb;
S.B = eye(3);

% ---- eigen-decomposition, power bound constant M -------------------------
[Vr, Dc] = eig(S.A);
S.V  = Vr;
S.mu = diag(Dc);
S.M  = cond(Vr);                     % ||A^k|| <= M*rho^k

% ---- eigen-directions C and eigen-coordinate map Pi ----------------------
% rows of inv(V) are left eigenvectors:  l_i*A = mu_i*l_i,  c_i^H = l_i
Lft  = inv(Vr);
kept = true(3,1);
Pi   = [];  S.dim = [];  S.mu_C = [];  S.Lam = {};
for i = 1:3
    if ~kept(i), continue; end
    li = Lft(i,:);  mui = S.mu(i);
    if abs(imag(mui)) < 1e-10                       % real eigenvalue -> d_i = 1
        a = real(li);
        Pi = [Pi; a];  S.dim(end+1) = 1;
        S.Lam{end+1} = real(mui);
    else                                            % complex pair -> d_i = 2
        a = real(li);  b = imag(li);
        Pi = [Pi; a; b];  S.dim(end+1) = 2;
        sg = real(mui);  om = imag(mui);
        S.Lam{end+1} = [sg -om; om sg];
        j = find(abs(S.mu - conj(mui)) < 1e-9 & kept);   % drop the conjugate
        kept(j(1)) = false;
    end
    S.mu_C(end+1) = mui;
    kept(i) = false;
end
S.Pi    = Pi;
S.r     = numel(S.dim);
S.alpha = min(svd(Pi));              % underline{alpha} = sigma_min(Pi)

% ---- input set and initial state set ------------------------------------
du   = 0.4;
S.U  = [zeros(3,1), du*[eye(3), -eye(3)]];    % |U| = 7, contains 0
S.rU = max(vecnorm(S.B*S.U));
S.Ibox = [-1 1];                              % I = [-1,1]^3
end
