%% May 2023 SCEGRAM visual search
clear all; close all; %clc;
result = getExperimentDetails();   
subjectName = result.subject_initials;
subjectId = regexp(subjectName, 'subj([0-9]+)\-.*', 'tokens');
subjectId = str2double(subjectId{1}{1});          
subjectAge = result.subject_age;
subjectGlasses = result.subject_glasses;
sessionNumber = result.session;
eyelink = result.eyelink; % whether to use eyelink
firstBlock = result.block2start; % which block to start the experiment on
ExperimentMode= result.ExperimentMode; % which exp mode: 0 exp mode; 1 test mode
monitorID = 0;

% confirm experimental details
fprintf('*************************************************\n');
fprintf('PHYSIOLOGY_EXPERIMENT Identification-Occlusion\n');
fprintf('Subject: %s (id %d) | %d years old, wears glasses: %d\n', ...
    subjectName, subjectId, subjectAge, subjectGlasses);
fprintf('Eyelink: %d\n', eyelink);
fprintf('Session #: %d\n', sessionNumber);
fprintf('\n');
fprintf('Experiment Start: %s\n', datestr(now));
fprintf('Experiment Mode %d, block %d:\n', ExperimentMode, firstBlock);
fprintf('*************************************************\n');
answer = input('Is the above information correct? [y/n]$ ','s');
if ~strcmpi(answer, 'y')
    error('Please rerun the experiment with correct data.');
end

%% setup eyetracker
if(eyelink)
    [eyelinkFile, success] = setupEyetrackerSG(monitorID, subjectName);
    display('eyetracker')
    display(eyelinkFile);
    if ~success
        eyelink = false;
    end
else
    eyelinkFile = '';
end

%% run actual experiment
mainPath='C:\Users\zlabe\OneDrive\Desktop\eye tracking\codes'; 

if ExperimentMode == 0
    images_per_block = 30; %defined by mengmi
elseif ExperimentMode == 1 %test mode
    images_per_block=6;
else
    error('Invalid occlusion type %d', ExperimentMode);
end


 SCEGram_visSearch(...
'subject_name', subjectName, 'subject_id', subjectId, ...
'subject_glasses', subjectGlasses, 'subject_age', subjectAge, ...
'session', sessionNumber, ...
'start_index', firstBlock, 'n_images_in_block', images_per_block, ...
'eyelink', eyelink, 'eyelink_file', eyelinkFile, ...
'monitor_ID', monitorID);


Screen('CloseAll');
display(['experiment completed!']);


