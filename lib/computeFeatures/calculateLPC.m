function lpcCoeff = calculateLPC(audio,order)
%CALCULATELPC Calcula los LPC de un audio
%   Detailed explanation goes here

% Window the speech segment using a Hamming window.
x1 = audio.*hamming(length(audio));

% Apply a pre-emphasis filter. The pre-emphasis filter is a highpass
% all-pole (AR(1)) filter.
preemph = [1 0.63];
x1 = filter(1,preemph,x1);

% Obtain the linear prediction coefficients. To specify the model order,
% use the general rule that the order is two times the expected number of
% formants plus 2. In the frequency range, [0,|Fs|/2], you expect three
% formants. Therefore, set the model order equal to 8. Find the roots of
% the prediction polynomial returned by lpc.
lpcCoeff = lpc(x1,order);

end







