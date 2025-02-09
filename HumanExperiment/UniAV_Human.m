function UniAV_Human(varargin)
    
    Screen('Preference', 'SkipSyncTests', 1);
    KbName('UnifyKeyNames');
    exp_params.monitor_id = 0; %1;  0: main monitor, select monitor for experiments
    exp_params.subject_name = 'yanhao';
    exp_params.subject_id = '';
    exp_params.subject_glasses = false;
    exp_params.subject_age = -1;
    exp_params.session = 1;
    exp_params.ImageDir = "dataset\image\";
    exp_params.AudioDir = "dataset\";
    exp_params.eyelink = 1;
    exp_params.eyelink_file = '';
    exp_params.fixation_threshold = 1.0;    % radius in visual angle degrees around fixation point to count as fixation
    exp_params.fixation_time = 0.5;         % time in seconds fixation must be maintained before trial starts
    exp_params.fixation_timeout = 10;       % number of seconds to wait at fixation point before asking about recalibration
    exp_params.fixation_size = 28;
    exp_params.fixation_width = 2;
    exp_params = parseArgs(exp_params, varargin{:});
    if ~exp_params.subject_name
        error('No subject_name given')
    end
    if ~exp_params.subject_id
        error('No subject_id given')
    end
    display('sucessful loading');
    params = struct();
    params.font_size=24;
    params.exit_key=KbName('Escape');
    params.STARTTRIALFIXATIONDELAY = 1.0;
    params.each_task_num_images = 3;
    params.each_bin_num_images = 1;
    params.each_task_num_cls = 1;
    params.each_cls_num_images = 1;
    params.text_color=0;
    params.trail_images=40;
    params.ARRAYPRESENTATIONTIME = .3;
    params.TARGETPRESENTATIONTIME = 1.5;
    params.TARGETARRAYDELAY = 1.0;
    params.MAXTRIALTIME = 2.0;
    params.WaitSearch = 20;         % wait for 30 seconds to go to the next trial if target not found
    params.TrialScore = [];
    params.TrialHistory = [20 20 20];
    params.colormode = 1;           % 0: grayscale target; 1: color image
    
    % fixation texture
    [window, windowRect] = Screen('OpenWindow', exp_params.monitor_id, 128);
    disp([windowRect(4) windowRect(3)]);
    [params.screenXpixels, params.screenYpixels] = Screen('WindowSize', window);
    params.window = window;
    params.window_rect = windowRect;
    params.ctrx = floor(windowRect(3)/2);
    params.ctry = floor(windowRect(4)/2);
    params.white = WhiteIndex(window);      % pixel value for white
    params.black = BlackIndex(window);      % pixel value for black
    params.gray = floor((params.white+params.black)/2);
    fixationMatrix = createFixationMatrix(exp_params.fixation_size, exp_params.fixation_width, params.gray, params.black);
    fixationScreen = Screen('maketexture', window, fixationMatrix);
    keyboards = GetKeyboardIndices;

    if exp_params.eyelink
        Eyelink('Initialize');
        Eyelink('Openfile', exp_params.eyelink_file);
        Eyelink('message', 'Test');
    end

    try
        HideCursor;
        [eyelinkFile, success] = setupEyetrackerSG(exp_params.monitor_id, exp_params.subject_name);
        display('eyetracker') 
        display(eyelinkFile);
        if ~success
            eyelink = false;
        end

        Screen('Screens');  % make sure all functions (SCREEN.mex) are in memory
        FlushEvents('keyDown');
    
        subjectDir = fullfile('subjects_SCEgram', exp_params.subject_name);
        if ~exist(subjectDir, 'dir')
            fprintf('Creating subject directory: %s\n', subjectDir);
            mkdir(subjectDir);
        else
            fprintf('Using existing subject directory %s\n', subjectDir);
        end
    
        jsonFiles = {'dataset/cond1/config.json', 'dataset/cond3/config.json', 'dataset/cond4/config.json', 'dataset/cond6/config.json'};
        tasklist = {1, 3, 4, 6};
        result = struct('x', {}, 'y', {}, 'area_class', {}, 'cls', {}, 'image', {}, 'audio', {}, 'task', {}, 'time', {});
    
        if exp_params.eyelink
            Eyelink('message','SCEGram');
        end
    
        for fileIdx = 1:length(jsonFiles)
            jsonFile = jsonFiles{fileIdx};
            task = tasklist{fileIdx};
    
            fid = fopen(jsonFile);
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);
            data = jsondecode(str);
            binSizes = unique([data.area_class]);

            for i = 1:3
                binSize = binSizes(i);
                binData = data([data.area_class] == binSize);

                % get all classes
                clsList = unique({binData.cls});
                indexToDelete = find(strcmp(clsList, 'hair drier'));
                if ~isempty(indexToDelete)
                    clsList(indexToDelete) = [];
                end
                numCls = length(clsList);

                % randomly choose n classes
                selectedClsIndices = randperm(numCls, params.each_task_num_cls);
                selectedCls = clsList(selectedClsIndices);

                % for each class, randomly choose n images
                for j = 1:length(selectedCls)
                    cls = selectedCls{j};
                    clsData = binData(strcmp({binData.cls}, cls));
                    
                    disp(clsData);
                    selectedIndices = randperm(length(clsData), params.each_cls_num_images);
                    
                    for k = 1:length(selectedIndices)
                        idx = selectedIndices(k);
                        result(end+1).area_class = binSize;
                        result(end).cls = cls;
                        result(end).image = strrep(clsData(idx).image, './', '');
                        result(end).audio = strrep(clsData(idx).audio, './', '');
                        result(end).task = task;
                    end
                end
            end
        end
        
        jsonFile = 'dataset/cond2/config.json';
        task = 2;
        fid = fopen(jsonFile);
        raw = fread(fid, inf);
        str = char(raw');
        fclose(fid);
        data = jsondecode(str);
        numPairs = length(data);
        selectedIndices = randperm(numPairs, params.each_task_num_images);
    
        % store the selected images and audios
        for i = 1:length(selectedIndices)
            idx = selectedIndices(i);
            result(end+1).area_class = data(idx).area_class;
            result(end).cls = data(idx).cls;
            result(end).image = strrep(data(idx).image, './', '');
            result(end).audio = strrep(data(idx).audio, './', '');
            result(end).task = task;
        end
    
        % audio calibration and validation
        audioCalibration(window, windowRect);
        audioValidation(window, windowRect);
        if exp_params.eyelink
            Eyelink('message','EXP_START');
        end
    
        line1 = 'Ready.';
        line2 = '\n Press SPACE key to begin.';
        line3 = '\n Press ESCAPE repeatedly to exit program.';
        line4 = '\n You will see an image and hear an audio clip, you need to locate the sounding object and use the mouse to click it.';
        line5 = '\n Before you find out the object, please do not move the mouse or click.';
        DrawFormattedText(window, [line1 line2 line3 line4 line5],'center', params.screenYpixels * 0.45, params.black);   
        Screen('Flip',window);    
        WaitSecs(0.2); 
        KbWait();
    
        randomPair = randperm(length(result));
    
        for i = 1:length(randomPair)
            if exp_params.eyelink
                Eyelink('message', ['TRIAL_ON: ' num2str(i)]);
                Eyelink('StartRecording');
            end
    
            if i == round(length(randomPair) / 2)
                line1 = 'Ready. You have completed a half of the experiment.';
                line2 = '\n You can have a break. When you are ready to begin the other half experiment,';
                line3 = '\n Press Enter key to begin the new audio calibration.';
                DrawFormattedText(window, [line1 line2 line3],'center', params.screenYpixels * 0.45, BlackIndex(window));   
                Screen('Flip',window);    
                WaitSecs(0.2); 
                KbWait();
                audioValidation(window, windowRect);
            end
    
            if exp_params.eyelink
                Eyelink('message','EXP_START');
            end
    
            img_path = exp_params.ImageDir + result(randomPair(i)).image;
            mid_folder = 'cond' + string(result(randomPair(i)).task) + '\audio';
            audio_path = exp_params.AudioDir + mid_folder + '\' + result(randomPair(i)).audio;
            image = imread(img_path);
            [audio, fs] = audioread(audio_path);
            player = audioplayer(audio, fs);
            
            % Clear screen and draw cross on the screen
            Screen('FillRect', window,  params.gray);
            HideCursor;
            if(exp_params.eyelink)
                exp_params = AwaitGoodFixation_natural('subjects_SCEGram',exp_params, params.window, params.window_rect, fixationScreen, curry(@WaitForKeyKeyboard, keyboards,0, params.exit_key));
            else
                Screen('fillrect',params.window,  params.black, [params.ctrx-14 params.ctry-1 params.ctrx+14 params.ctry+1]);
                Screen('fillrect',params.window,  params.black, [params.ctrx-1 params.ctry-14 params.ctrx+1 params.ctry+14]);
                Screen('Flip', window);
                WaitSecs(params.STARTTRIALFIXATIONDELAY);
            end

            Screen('FillRect', window,  params.gray);
            image = imresize(image, [1280, 1024]);
            imageTexture = Screen('MakeTexture', window, image);
            Screen('DrawTexture', window, imageTexture, [], [], 0);
            Screen('Flip', window);
            playblocking(player);
            ShowCursor;
            clicked = false;
            startTime = GetSecs();
            while GetSecs() - startTime < 5
                [x, y, buttons] = GetMouse(window);
                if any(buttons)
                    clicked = true;
                    result(randomPair(i)).time = GetSecs() - startTime;
                    % record the location of the click
                    result(randomPair(i)).x = x;
                    result(randomPair(i)).y = y;
                    if exp_params.eyelink
                        Eyelink('message', ['LOCATION FOUND']);
                    end
                    break;
                end
            end
            
            if exp_params.eyelink
                Eyelink('message', ['TRIAL_OFF: ' num2str(i)]);
                Eyelink('StopRecording');
            end 
            result(randomPair(i)).time = GetSecs() - startTime;
        end
        
        p = sprintf('result.txt', task);
        fileID = fopen(p, 'w');
        fprintf(fileID, 'x\ty\timage\taudio\tgt_box\tbin_size\ttask\ttime\n');
        for i = 1:length(randomPair)
            fprintf(fileID, '%d\t%d\t%s\t%s\t%d\t%d\t%d\n', result(i).x, ...
                result(i).y, result(i).image, result(i).audio, ...
                result(i).area_class, result(i).task, result(i).time);
        end
    
        if exp_params.eyelink
            Eyelink('message', 'EXPERIMENT_END');
            Eyelink('StopRecording');
            localEyelinkFile = getEyelinkFilepath_SG('subjects_SCEgram','CVS_MM_', ...
                exp_params.subject_name);
            status = Eyelink('closefile');
            if status ~= 0
                fsprintf('closefile error, status: %d', status);
            end
            Eyelink('ReceiveFile', exp_params.eyelink_file, localEyelinkFile);
        end
    
        Screen('CloseAll');
        display('experiment completed!');
    catch
        if exp_params.eyelink
            Eyelink('message', 'EXPERIMENT_END');
            Eyelink('StopRecording');
            localEyelinkFile = getEyelinkFilepath_SG('subjects_SCEgram','CVS_MM_', ...
            exp_params.subject_name);
            status = Eyelink('closefile');
            if status ~= 0
                fsprintf('closefile error, status: %d', status);
            end
            Eyelink('ReceiveFile', exp_params.eyelink_file, localEyelinkFile);
        end
        Screen('CloseAll');
        clear Screen;
        ShowCursor;
        display('Error but file has been saved');
        err = lasterror; disp(err.message); for stack_ind = 1:length(err.stack); disp(err.stack(stack_ind));  end
    end
end


% subfunction
function [keyCode, keyTime] = WaitForKeyKeyboard(...
    keyboards, duration, exit_keys)
keyCode = [];
keyTime = [];
startTime = GetSecs();
b = 0;
while ~b && (GetSecs() - startTime < duration || duration == 0)
    [key_down, keyTime, keyCode] = KbCheck(keyboards(1));
    if any(key_down)
        b = 1;
    end
    if any(keyCode(exit_keys))
        error('Exit key pressed!');
    end
end
if GetSecs() - startTime < duration
    WaitSecs(startTime + duration - GetSecs());
end
end