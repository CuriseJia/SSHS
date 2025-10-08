clear all; close all;

while true
    inputs = experimentInputs();
    subjectName = inputs.subjectName;
    subjectId = inputs.subjectId;
    subjectAge = inputs.subjectAge;
    
    subjectGlasses = inputs.subjectGlasses;
    eyelink = inputs.eyelink;
    monitorID = 0;

    % Input verification
    fprintf('****************************************************************\n');
    fprintf('Please confirm that the provided information is correct:\n');
    fprintf('Subject Name: %s \n', subjectName);
    fprintf('Subject Age: %d \n', subjectAge);
    fprintf('Glasses: %d \n', subjectGlasses);
    fprintf('Eyelink: %d \n', eyelink);
    fprintf('****************************************************************\n');
    reply = input('Are the provided information correct [y/n]?', 's');

    if strcmpi(reply, 'y')
        break;
    else
        fprintf('Key in the information again');
        continue;
    end
end

% Set up eye tracker
if eyelink
    [eyelinkFile, sucess] = setupEyeTracker(monitorID, subjectName, '', true, [], []);
    if ~sucess
        eyelink=false;
        fprintf('Set up of Eyetracker has failed, proceeding without Eyelink\n');
    end
else
    eyelinkFile = '';
end

% Call experiemnt process
display(subjectId)
experimentProcess('subjectName', subjectName, 'subjectId', subjectId, 'subjectAge', subjectAge, 'subjectGlasses', subjectGlasses, 'eyelink', eyelink, 'monitorID', monitorID, 'eyelinkFile', eyelinkFile);

Screen("CloseAll");
fprintf("Experiment Completed");