function [eyelinkFile, sucess] = setupEyeTracker(monitorId, subjectName, eyelinkFile, initialise, window, window_rect)
    Eyelink('Shutdown');
    sucess = false;
    Screen('Preference', 'SkipSyncTests', 1);
    
    if initialise
        [window, window_rect]=Screen('OpenWindow' ,monitorId, 128);
        Priority(MaxPriority(window));

        % Create Eyelink File to store fixation data
        c = clock;
        eyelinkFile = [subjectName(end-1:end), ...
            num2str(mod(c(3),10),'%0.1d'), ...
            num2str(c(4),'%0.2d'), ...
            num2str(c(5),'%0.2d') '.edf'];
    end
    
    % Processes to Calibrate Eye tracker
    % Initialise default settlings for Eyelink 
    eyeLink = EyelinkInitDefaults(window);
    % Initialise connection with the Eyelink
    if ~EyelinkInit(0,1)
        fprintf("Unsecessful eyelink initialisation")
        cleanup;
        return;
    end
    
    % [~, vs]=Eyelink('GetTrackerVersion');
    % formatStr = "Experiment running on eyetracker version" + vs;
    % fprintf(formatStr);

    SCREEN_SIZE_CM = [55, 30.5]; %cm (current resolution of lab monitor is 1920x 1080)
    % SCREEN_SIZE_CM = [31.26, 22.12]; %For macbook
    SCREENPHYSICALCOORDINATES = [-SCREEN_SIZE_CM(1,1)/2, SCREEN_SIZE_CM(1,2)/2,...
    SCREEN_SIZE_CM(1,1)/2, -SCREEN_SIZE_CM(1,2)/2]*10; % NEEDS TO BE IN MM
    Eyelink('command','screen_phys_coords=%s',num2str(SCREENPHYSICALCOORDINATES))% : to set the screen size in mm
    % Setting up eyetracker commands

    Eyelink('Command', 'screen_pixel_coords = %d %d %d %d', window_rect(1),window_rect(2),window_rect(3)-1,window_rect(4)-1);
    Eyelink('Message', 'DISPLAY_COORDS %d %d %d %d', window_rect(1),window_rect(2),window_rect(3)-1,window_rect(4)-1);
    Eyelink('Command', 'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT');
    Eyelink('Command', 'link_event_filter = LEFT,RIGHT,FIXATION,BUTTON');

    % Obtain fixation data from eyelink device
    Eyelink('Command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');

    % Create edf file to store fixation data
    Eyelink('Openfile', eyelinkFile);

    % Calibrate the eye tracker
    result = EyelinkDoTrackerSetup(eyeLink);

    % Perform drift correction
    EyelinkDoDriftCorrection(eyeLink);

    % Start recording fixation data and wait for a few seconds before we start displaying
    Eyelink('StartRecording');
    WaitSecs(0.1);
    % mark zero-plot time in data file  
    Eyelink('Message', 'DISPLAY_ON');	% message for RT recording in analysis
    Eyelink('Message', 'SYNCTIME');
    
    sucess = true;
    if initialise
        Screen('CloseAll');
    end
    return;
end