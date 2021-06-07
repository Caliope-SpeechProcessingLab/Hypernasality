function VLHR = calculateVLHR(audioIn,fs,varargin)
% -------------------------------------------------------------------------
% function VLHR = calculateVLHR(audioIn,fs,varargin)
% -------------------------------------------------------------------------
% DESCRIPTION:
% Calculated Voice Low Tone to High Tone Ratio (VLHR) as described in:
% Guo-She Lee, Ching-Ping Wang, C. C. H. Yang and T. B. J. Kuo, "Voice low
% tone to high tone ratio: a potential quantitative index for vowel [a:]
% and its nasalization," in IEEE Transactions on Biomedical Engineering,
% vol. 53, no. 7, pp. 1437-1439, July 2006, doi: 10.1109/TBME.2006.873694.
% -------------------------------------------------------------------------
% INPUTS:
% - audioIn: audio data
% - fs: sample frequency
% - varargin: extra optional params
%           - 'cutoff',cutoff: Cutoff frequency
% -------------------------------------------------------------------------
% OUTPUTS:
% - VLHR: VLHR feature in db.
% -------------------------------------------------------------------------
% Default values
defaultCutoffFreq = 600;
cutoofLow = 65;
cutoofHigh = 8000;

% Input Parser
p = inputParser;
validAudioIn = @(x) ~isempty(audioIn);
validFs = @(x) fs > 0;
addRequired(p,'audioIn',validAudioIn);
addRequired(p,'fs',validFs);
addOptional(p,'cutoff',defaultCutoffFreq,@(x) x>cutoofLow && x<cutoofHigh);
parse(p,audioIn,fs,varargin{:});

audio = p.Results.audioIn;
fs = p.Results.fs;
cuttoffFreq = p.Results.cutoff;

% VLHR
[ltas,freqAxis] = iosr.dsp.ltas(audioIn,fs,'units','none');

LFP = sum( ltas(logical((freqAxis>cutoofLow).*(freqAxis<cuttoffFreq))) );
HFP = sum( ltas(logical((freqAxis>cuttoffFreq).*(freqAxis<cutoofHigh))) );

VLHR = 10*log10(LFP/HFP);

end % function