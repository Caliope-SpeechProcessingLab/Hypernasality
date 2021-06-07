function stcOut = createNewPatient(patientName)
% -------------------------------------------------------------------------
% function stcOut = createNewPatient(sName)
% -------------------------------------------------------------------------
% DESCRIPTION:
% This functions takes audios generated from a patient and saves them as an
% struct for futher analysis.
% -------------------------------------------------------------------------
% INPUTS:
% - patientName: Patient name e.g., patientStc = createNewPatient('fem31017')
% -------------------------------------------------------------------------
% OUTPUTS: 
% - stcOut: Output struct with:
%           - stcOut.info.class: Patient class (hypernasal or control)
%           - stcOut.info.id: Patient id
%           - stcOut.info.age: Patient age
%           - stcOut.info.fs: Patient sample frequency of audio data
%           - stcOut.audios.xxx: Audios generated
% -------------------------------------------------------------------------
stcOut = struct();

sNames = dir(fullfile('audio',patientName));
        
% Patient info
stcOut.info.class = patientName(1:3);
stcOut.info.id = patientName(6:end);
stcOut.info.age = patientName(4:5);

% Save all audio generated from ASICA app
for m = 1:length(sNames)
    if startsWith(sNames(m).name,"record",'IgnoreCase',true)
        try 
            nameAudio = (sNames(m).name(1:end-4));  % delete '.wav'
            % fix names
            nameAudio = strrep(nameAudio,'-','_');  % replace '-' for '_'
            nameAudio = strrep(nameAudio,'á','a');  
            nameAudio = strrep(nameAudio,'ó','o');  
            nameAudio = strrep(nameAudio,'ñ','n');  

            if ~endsWith(nameAudio,["Consigna","Despedida"])
                [y,fs] = audioread(fullfile(sNames(m).folder,sNames(m).name));
                stcOut.audio.(nameAudio) = y;
                stcOut.info.fs = fs;
            end % if 
        catch
            disp(['Error loading data: ' nameAudio ' ' patientName]);
        end % try
    end % if
end % for m
   
end % function







