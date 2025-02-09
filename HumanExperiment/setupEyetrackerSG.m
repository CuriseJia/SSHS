function [eyelinkFile, success] = setupEyetrackerSG(monitorID, subj)
%SETUPEYETRACKER Summary of this function goes here
Screen('Preference', 'SkipSyncTests', 1); % RJ: Take note of skip sync test
InitializePsychSound;
try
    % cycle eyelink shutdown/initialize for a bit
    Eyelink('Shutdown');
   
    [window,window_rect]=Screen('OpenWindow',monitorID,128, []); %, [marginsPix,0,scrRes.width-marginsPix,scrRes.height]);

   %%%[window,window_rect]=Screen('OpenWindow',monitorID,128, [marginsPix,0,scrRes.width-marginsPix,scrRes.height]);

   %%%%%%%%%%
    Screen('flip',window);

    Priority(MaxPriority(window,'WaitBlanking'));

    c = clock;
    %     eyelinkFile = [subj(end-2:end), ...
    %         num2str(mod(c(3),10),'%0.1d'), ...
    %         num2str(c(4),'%0.2d'), ...
    %         num2str(c(5),'%0.2d') '.edf'];
    eyelinkFile = [subj(end-1:end), ...
        num2str(mod(c(3),10),'%0.1d'), ...
        num2str(c(4),'%0.2d'), ...
        num2str(c(5),'%0.2d') '.edf'];
    fprintf('eyelink file: %s\n',eyelinkFile);
    status = calibrateEyelink(window,window_rect,eyelinkFile); %, resWidth);

    success = status == 0;
    if(~success)
        fprintf('**************************\n');
        answer = input('Error during calibration.. continue without eyelink? [y/n]','s');
        if ~strcmpi(answer, 'y')
            error('Please rerun the experiment');
        else
            fprintf('Continuing without eyelink...');
            Eyelink('Shutdown');
        end
    end

    ShowCursor;
    Screen('CloseAll');
catch err
    disp(getReport(err, 'extended'));
    Screen('CloseAll');
    ShowCursor;
    throw(err);
end
end

