function [param, audioCalibrationNeeded] = checkFixationAccuracy(param, window, window_rect)
    accuracy = false;
    count = 0;
    audioCalibrationNeeded = 0;
    while ~accuracy
        Screen('fillrect',param.window,  param.black, [param.ctrx-param.half_fix_size param.ctry-param.half_fix_width param.ctrx+param.half_fix_size param.ctry+param.half_fix_width]);
        Screen('fillrect',param.window,  param.black, [param.ctrx-param.half_fix_width param.ctry-param.half_fix_size param.ctrx+param.half_fix_width param.ctry+param.half_fix_size]);
        Screen('Flip', window);
        accuracy = getAccuracy(window_rect, param);

        if ~accuracy
            fprintf('INACCURATE\n');
            % line1 = 'Unable to detect accuate fixation. Select recalibration option [d/c/n]?';
            if ~count
                line1 = 'Unable to detect accuate fixation. Conducting Drift Correction. Press any Key to Begin';
            else
                line1 = 'Unable to detect accuate fixation. Conducting Eye Tracking Calibration. Press any Key to Begin';
            end
            DrawFormattedText(window, line1, 'center', 'center', param.black);
            Screen('flip', window);
            KbWait();
            % [~, ~, keyCode] = KbCheck;

            if ~ count
                Eyelink('message', 'DriftCorrection')
                eyeLink = EyelinkInitDefaults(window);
                EyelinkDoDriftCorrection(eyeLink);
            else
                audioCalibrationNeeded = 1;
                Eyelink('message', 'Calibration')
                eyeLink = EyelinkInitDefaults(window);
                EyelinkDoTrackerSetup(eyeLink);
                EyelinkDoDriftCorrection(eyeLink);
            end

            Eyelink('StartRecording');
            WaitSecs(0.1);

            count = count + 1;
        end
    end
end


function accuracy = getAccuracy(window_rect, param)
    fix_x = window_rect(3)/2;
    fix_y = window_rect(4)/2;
    fix_threshold = param.fix_treshold;
    fix_timing = param.fix_timing;
    fix_timeout = param.fix_timeout;

    startTime = GetSecs;
    accuracy = 0;

    latest_time = -1;
    first_time = -1;

    while (GetSecs-startTime < fix_timeout)
        fixation = Eyelink('NewestFloatSample');
        
        if isstruct(fixation)
            gx = max(fixation.gx);
            gy = max(fixation.gy);

            fix_error = sqrt(((fix_x - gx))^2 + ((fix_y - gy))^2);

            % Count the amount of time a fixation is held within a given threshold
            if fix_error <= fix_threshold
                latest_time = GetSecs;
                if first_time == -1
                    first_time = latest_time;
                end
            % If the threshold is broken, reset the process
            else
                latest_time = -1;
                first_time = -1;
            end

            % If an accurate fixation is held for a predetermined timing, break from loop
            if latest_time-first_time > fix_timing
                accuracy = 1;
                break;
            end
        % Incase the user blinks or looks away and thus a structure was not obtained
        elseif GetSecs - latest_time > 0.05
            latest_time = -1;
            first_time = -1;
        end 
    end

    display(gx);
    display(gy);
end