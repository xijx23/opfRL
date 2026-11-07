function out = opf_sirl_collect(results)
%OPF_SIRL_COLLECT Pack objective, status and MATPOWER input tables for Python.
%   Only the input columns of bus/gen/branch are kept (13/21/13), so the tables
%   are comparable between runopf results and a plain mpc struct.
out = struct();
out.f = double(results.f);
if isfield(results, 'success')
    out.success = double(results.success);
else
    out.success = NaN;
end
widths = struct('bus', 13, 'gen', 21, 'branch', 13);
names = {'bus', 'gen', 'branch', 'gencost'};
for k = 1:numel(names)
    nm = names{k};
    if isfield(results, nm) && isnumeric(results.(nm))
        t = results.(nm);
        if isfield(widths, nm)
            t = t(:, 1:min(size(t, 2), widths.(nm)));
        end
        out.(nm) = double(t);
    end
end
end
