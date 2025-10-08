clear all; close all; clc;

%% Plot the number of fixation distribution.

%% Add necessary paths and directory information
addpath('./');
taskdirectory = './Subjects';

subjname = 'subj04-yh';
edflist = dir([taskdirectory '/' subjname '/SSHS_' subjname '-*.edf']);
save_path = fullfile(taskdirectory, subjname);

%% Initialize storage for CDF data from all trials
% allDurations = [];

%% Process each EDF file
for i = 1:length(edflist)
    disp(['Processing file ', num2str(i), ' of ', num2str(length(edflist))]);

    % Load .edf file
    edf0 = Edf2Mat([taskdirectory '/' subjname '/' edflist(i).name]);
    eventlist = edf0.RawEdf.FEVENT;
    edf_name = strsplit(edflist(i).name, '.');

    edf_name = ['eventlist_', edf_name{1}];
    save([save_path, '/', edf_name, '.mat'], 'eventlist');
    
    disp(['EDF is saved in: ', edf_name, '.mat']);
    
    load_data = load([save_path, '/', edf_name, '.mat']);
    disp(fieldnames(load_data));


end




