function [formantsOut, bwOut] = calculateFormants(audioIn,fs)
% -------------------------------------------------------------------------
% [formantsOut, bwOut] = calculateFormants(audioIn,fs)
% -------------------------------------------------------------------------
% DESCRIPTION:
% Calculated Voice Low Tone to High Tone Ratio (VLHR) as described in:
% https://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
% -------------------------------------------------------------------------
% INPUTS:
% - audioIn: audio data
% - fs: sample frequency
% -------------------------------------------------------------------------
% OUTPUTS:
% - formantsOut: formants frequency
% - bwOut: formants bandwidth
% -------------------------------------------------------------------------
% Two common preprocessing steps applied to speech waveforms before linear
% predictive coding are windowing and pre-emphasis (highpass) filtering.

% Window the speech segment using a Hamming window.
x = audioIn;
x1 = x.*hamming(length(x));

% Apply a pre-emphasis filter. The pre-emphasis filter is a highpass
% all-pole (AR(1)) filter.
preemph = [1 0.63];
x1 = filter(1,preemph,x1);

% Obtain the linear prediction coefficients. To specify the model order,
% use the general rule that the order is two times the expected number of
% formants plus 2. In the frequency range, [0,|Fs|/2], you expect three
% formants. Therefore, set the model order equal to 8. Find the roots of
% the prediction polynomial returned by lpc.
A = lpc(x1,16);
rts = roots(A);

% Because the LPC coefficients are real-valued, the roots occur in complex
% conjugate pairs. Retain only the roots with one sign for the imaginary
% part and determine the angles corresponding to the roots.
rts = rts(imag(rts)>=0);
angz = atan2(imag(rts),real(rts));

% Convert the angular frequencies in rad/sample represented by the angles
% to hertz and calculate the bandwidths of the formants.

% The bandwidths of the formants are represented by the distance of the
% prediction polynomial zeros from the unit circle.
[frqs,indices] = sort(angz.*(fs/(2*pi)));
bw = -1/2*(fs/(2*pi))*log(abs(rts(indices)));

% Use the criterion that formant frequencies should be greater than 90 Hz
% with bandwidths less than 400 Hz to determine the forma
nn = 1;
for kk = 1:length(frqs)
    if (frqs(kk) > 90 && bw(kk) < 400)
        formants(nn) = frqs(kk);
        bwformants(nn) = bw(kk);
        nn = nn+1;
    end
end

% Hay algunos casos en los que no encuentra los formates por la condición
% del ancho de banda menor de 400Hz. En ese caso ampliamos esa condición
if length(formants) < 3
    for kk = 1:length(frqs)
        if ( frqs(kk) > 90 && bw(kk) < 450 && ~any(ismember(formants,frqs(kk))) )
            formants(end+1) = frqs(kk);
            bwformants(end+1) = bw(kk);
        end
    end
end % while 

if length(formants) < 3
    for kk = 1:length(frqs)
        if ( frqs(kk) > 90 && bw(kk) < 500 && ~any(ismember(formants,frqs(kk))) )
            formants(end+1) = frqs(kk);
            bwformants(end+1) = bw(kk);
        end
    end
end % while 

if length(formants) < 3
    for kk = 1:length(frqs)
        if ( frqs(kk) > 90 && bw(kk) < 550 && ~any(ismember(formants,frqs(kk))) )
            formants(end+1) = frqs(kk);
            bwformants(end+1) = bw(kk);
        end
    end
end % while 

if length(formants) < 3
    for kk = 1:length(frqs)
        if ( frqs(kk) > 90 && bw(kk) < 600 && ~any(ismember(formants,frqs(kk))) )
            formants(end+1) = frqs(kk);
            bwformants(end+1) = bw(kk);
        end
    end
end % while 

[formants, idx] = sort(formants,'ascend');
bwformants = bwformants(idx);

try
    formantsOut = formants(1:3);
    bwOut = bwformants(1:3);
catch
    formantsOut = [formants(1:2),NaN];
    bwOut = [bwformants(1:2),NaN];
end % try

end % function


