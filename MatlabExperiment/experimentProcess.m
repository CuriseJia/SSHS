 function experimentProcess(varargin)
    rand('state', sum(100*clock)); 
    param = struct();

    param.subjectName = '';
    param.subjectAge = -1;
    param.subjectId = 1;
    param.subjectGlasses = false;
    param.eyelink = 0;
    param.monitorID = 0;
    param.CSIZE=256;

%     params.each_task_num_images = 60;
%     params.each_bin_num_images = 20;
%     params.each_task_num_cls = 10;
%     params.each_cls_num_images = 2;

    params.each_task_num_images = 30; 
    params.each_bin_num_images = 10;
    params.each_task_num_cls = 10;
    params.each_cls_num_images = 1;

    % Monitor Parameters
    param.monitor_width = 0.53;
    param.monitor_height = 0.32;
    param.monitor_depth = 0.76;
    param.monitor_half_width = param.monitor_width/2;
    param.monitor_half_height = param.monitor_height/2;
    
    % Fixation paramaters
    param.eyelinkFile = '';
    % % % param.fix_treshold = 1.0;
    param.fix_treshold = 80;
    param.fix_timing = 0.5;
    param.fix_timeout = 5;

    % Experiment timings
    param.search_time = 20;

    param.fix_size = 28;
    param.fix_width = 2;

    param.half_fix_size = floor(param.fix_size/2);
    param.half_fix_width = floor(param.fix_width/2);

    param.font_size=24;
    param.colourmode = 1;

    param.STARTTRIALFIXATIONDELAY = 1.0;
    param.TARGETPRESENTATIONTIME = 1.5;
    param.AUDIOPRESENTATIONTIME = 4.0;
    param.STIMULILOADDELAY = 1.0;
    param.WaitSearch = 20;
    param.colourmode = 1;

    param.circleLocationsCalib = [1, 1; 1, 3; 2, 1; 2, 3; 3, 1; 3, 3];
    param.circleLocationsCalibValid = [1, 1; 1, 2; 1, 1; 1, 2; 1, 1; 1, 2;];
    
    KbName('UnifyKeyNames'); % Sync key presses across different devices
    param.exit_key = KbName('Escape');

    param = parseInputs(param, varargin{:});
    % Possibly do another verification for subject name and id?

    param.theta = zeros(2, 3);
    
    distractor_list = {'gaussian_noise_image.jpg','black_image.jpg'};
    jsonFiles = {'dataset/config1.json', 'dataset/config2.json', 'dataset/config3.json', 'dataset/config6.json'};
    tasklist = {1, 2, 3, 6};
    result = struct('x', {}, 'y', {}, 'object_size', {}, 'category', {}, 'image', {}, 'audio', {}, 'task', {}, 'time', {}, 'gt_box', {});
    temp = struct('x', {}, 'y', {}, 'object_size', {}, 'category', {}, 'image', {}, 'audio', {}, 'task', {}, 'gt_box', {}, 'time', {});
    
    for fileIdx = 1:length(jsonFiles)
        jsonFile = jsonFiles{fileIdx};
        task = tasklist{fileIdx};
        fid = fopen(jsonFile);
        raw = fread(fid, inf); 
        str = char(raw');
        fclose(fid);
        data = jsondecode(str);
    
        % 获取所有的 bin_size
        binSizes = unique([data.object_size]);
    
        % 对于每个 bin_size
        for i = 1:3
            binSize = binSizes(i);
            binSize = strcat('size', binSize);
            check = {data.object_size};
            mask = strcmp(check, binSize);
            binData = data(mask);
            
            % 获取所有的 category
            clsList = unique({binData.category});
            numCls = length(clsList);
            
            % 随机选择 m 个不同的 category
            selectedClsIndices = randperm(numCls, params.each_task_num_cls);
            selectedCls = clsList(selectedClsIndices);
            
            % 对于每个选定的 cls，随机选择 n 组 image/audio/gt_box
            for j = 1:length(selectedCls)
                category = selectedCls{j};
                clsData = binData(strcmp({binData.category}, category));
                
                selectedIndices = randperm(length(clsData), params.each_cls_num_images);
                
                for k = 1:length(selectedIndices)
                    idx = selectedIndices(k);
                    result(end+1).object_size = binSize;
                    result(end).category = category;
                    % Check
                    result(end).image = strrep(clsData(idx).image_id, './', '');
                    result(end).audio = strrep(clsData(idx).audio, './', '');
                    result(end).task = task;
                    result(end).gt_box = clsData(idx).gt_box;
                    result(end).center = clsData(idx).unity_point;
                end
            end
        end
    end
    
    jsonFile = 'dataset/config4.json';
    task = 4;
    fid = fopen(jsonFile);
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    data = jsondecode(str);
    numPairs = length(data);
    selectedIndices = randperm(numPairs, params.each_task_num_images);

    % 存储选择的 image/audio 对
    for i = 1:length(selectedIndices)
        idx = selectedIndices(i);
        result(end+1).object_size = data(idx).object_size;
        result(end).category = data(idx).category;
        % Check
        result(end).image = strrep(data(idx).image_id, './', '');
        result(end).audio = strrep(data(idx).audio, './', '');
        result(end).task = task;
        result(end).gt_box = data(idx).gt_box;
        result(end).center = data(idx).unity_point;
    end
    
    randomPair = randperm(length(result));

    save_dir = fullfile('Subjects', param.subjectName);
    
    if ~exist(save_dir, 'dir')
        mkdir(save_dir)
        fprintf('New directory has been created')
    else
        fprintf('Using existing directory')
    end

    Screen('Preference', 'SkipSyncTests', 1);
    Screen('Screens');

    FlushEvents('keyDown');
    HideCursor;

    [window, window_rect]=Screen('OpenWindow', param.monitorID, 128);
    [param.screenXpixels, param.screenYpixels] = Screen('WindowSize', window);
    
    param.window = window;
    param.window_rect = window_rect;
    param.ctrx = floor(window_rect(3)/2);
    param.ctry = floor(window_rect(4)/2);
    param.white = WhiteIndex(window);
    param.black = BlackIndex(window);
    param.gray = floor((param.white+param.black)/2);
    param.quad_width = param.window_rect(3) / 3;
    param.quad_height = param.window_rect(4) / 3;

    priority_level = MaxPriority(window);
    Priority(priority_level);

    Screen('TextSize', window, param.font_size);
    InitializePsychSound(1);

    % Setup TCP connection with unity for dual communication (MATLAB acts as a client)
    tcpClient = tcpclient('127.0.0.1', 55001, 'Timeout', 3);

    timeCollection = [];

    % Begin experiment
    line1='Ready.';
    line2='\n Press ANY KEY to begin, or move to next trial.';
    line3 ='\n Press ESCAPE repeatedly to exit program.';   
    DrawFormattedText(window, [line1 line2 line3],'center', param.screenYpixels * 0.45, param.black);   
    Screen('Flip',window);    
    WaitSecs(0.2); 
    KbWait();

    if param.eyelink
        Eyelink('message', 'Start_Exp')
    end

    for t=1 : length(randomPair)
        trial = result(randomPair(t));

        % Perform Audio Calibration
        if t==1
            param = audioProcess(tcpClient, param, t);
        end

        if param.eyelink
            Eyelink('message', ['Fixation Screen; Trial: ' num2str(t)]);
        end

        if (param.eyelink)
            [param, audioCalibrationNeeded] = checkFixationAccuracy(param, window, window_rect);
            
        else
            Screen('fillrect',param.window,  param.black, [param.ctrx-param.half_fix_size param.ctry-param.half_fix_width param.ctrx+param.half_fix_size param.ctry+param.half_fix_width]);
            Screen('fillrect',param.window,  param.black, [param.ctrx-param.half_fix_width param.ctry-param.half_fix_size param.ctrx+param.half_fix_width param.ctry+param.half_fix_size]);
            Screen('Flip', window);
            WaitSecs(param.STARTTRIALFIXATIONDELAY);
        end
        
        % Clear Screen
        Screen('FillRect', window,  param.gray);

        if trial.task == 4
            selectedImage = distractor_list{randi(length(distractor_list))};
            trial.image = string(selectedImage);
            img_path = [fullfile(fileparts(mfilename('fullpath')), 'dataset', filesep)] + string(selectedImage);
        else
            img_path = [fullfile(fileparts(mfilename('fullpath')), 'dataset', filesep, 'image', filesep)] + string(trial.image);
        end
        image = imread(img_path);
        stimuli_img = imresize(image, [param.screenYpixels, param.screenXpixels]);

        audioCoords = trial.center;
        
        audioPath = [fullfile(fileparts(mfilename('fullpath')), 'dataset', filesep, 'audio', filesep)] + string(trial.audio);

        % Display stimuli image
        textures = Screen('MakeTexture', window, stimuli_img);
        Screen('DrawTexture', window, textures, window_rect);

        if param.eyelink
            Eyelink('message', ['TRIAL_ON: ' num2str(t) 'image: ' result(randomPair(t)).image]);
        end

        translatedAudioCoords = computeTranslatedCoordinates(audioCoords, param.theta);

        Screen('Flip', window);

        timeCollection = startAudio(tcpClient, true, {translatedAudioCoords}, {audioPath}, timeCollection);
        
        t_start = GetSecs();
        ShowCursor;
        stopflag = 0;
        
        while stopflag==0
            [x, y, buttons] = GetMouse(window);
            t_now = GetSecs();

            x = int32(x);
            y = int32(y);

            if buttons(1) == 1
                Beeper('high',1,0.2);
                if param.eyelink
                    Eyelink('message', 'TARGET FOUND');
                end
                fprintf('Target found!\n');
                stopflag = 1;
                temp(t).time = t_now - t_start;
                % 记录点击位置
                temp(t).x = x;
                temp(t).y = y;
                temp(t).object_size = string(trial.object_size);
                temp(t).category = trial.category;
                temp(t).image = string(trial.image);
                temp(t).audio = string(trial.audio);
                temp(t).gt_box = trial.gt_box;
                temp(t).task = trial.task;
                break;
            end

            if (t_now - t_start) > param.WaitSearch
                Beeper('low',1,0.2);
                if param.eyelink
                    Eyelink('message', 'TIME EXCEED');
                end
                fprintf('Time exceed!\n');
                stopflag = 1;
                temp(t).time = t_now - t_start;
                % 记录点击位置
                temp(t).x = x;
                temp(t).y = y;
                temp(t).object_size = string(trial.object_size);
                temp(t).category = trial.category;
                temp(t).image = string(trial.image);
                temp(t).audio = string(trial.audio);
                temp(t).gt_box = trial.gt_box;
                temp(t).task = trial.task;
                fprintf('Target found!\n');
                break;
            end

            [~, ~, keyCode] = KbCheck;
            if keyCode(param.exit_key)
                fprintf('ESC pressed!\n');
                Beeper('low',1,0.2);          
                break;
            end
        end

        if stopflag == 0
            if param.eyelink
                Eyelink('message', 'EXPERIMNET_STOPPED');
            end
            break;
        end

        if param.eyelink
            Eyelink('message', ['TRIAL_OFF: ' num2str(t) 'image: ' result(randomPair(t)).image]);
        end   
        HideCursor;

        endAudio(tcpClient);
    end

    p = sprintf('result.txt');
    fileID = fopen(p, 'w');
    fprintf(fileID, 'x\ty\timage\taudio\tcategory\tobject_size\tgt_box\ttask\ttime\n');
    for i = 1:length(temp)
%         display(temp(i).gt_box);
%         fprintf(fileID, '%d\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n', temp(i).x, ...
%             temp(i).y, temp(i).image, temp(i).audio, temp(i).category, temp(i).object_size, temp(i).gt_box(1), temp(i).gt_box(2), temp(i).gt_box(3), temp(i).gt_box(4), temp(i).task, temp(i).time);
    end

    if param.eyelink
        Eyelink('message', 'EXPERIMENT_END');
    end

    line1 ='Thanks for taking part in the experiment!';   
    DrawFormattedText(window, line1,'center', param.screenYpixels * 0.5, param.black);
    Screen('flip',window);

    if param.eyelink
        file_path = getEyelinkFilePath('/Subjects', 'SSHS_', param.subjectName);
        full_file_path = [fullfile(fileparts(mfilename('fullpath')), file_path)]; 
        status = Eyelink('closefile');
        if status~=0
            fprintf('Error closing file\n')
        end
        Eyelink('ReceiveFile', param.eyelinkFile, full_file_path);
    end

    WaitSecs(2);
    meanTime = mean(timeCollection);
    fprintf("Mean TCP delay: %.2f ms\n", meanTime);
    clear tcpClient;
    Screen('CloseAll');

end


function param = parseInputs(param, varargin)
    for i=1:2:length(varargin)
        inputKey = cell2mat(varargin(i));
        inputVal = lower(cell2mat(varargin(i+1)));

        if strcmp(inputVal, '1') || strcmp(inputVal, '0')
            inputVal = str2num(inputVal);
        end

        if isfield(param, inputKey)
            param = setfield(param, inputKey, inputVal);
        else
            error('Invalid Input: %s', inputKey);
        end
    end
end

function afflineParams = computeAfflineParameters(X_MATLAB, Y_MATLAB, X_UNITY, Y_UNITY)
    n = length(X_MATLAB);
    MATLAB = zeros(n*2, 6);
    UNITY = zeros(n*2, 1);

    for i=1:n
        xmat = X_MATLAB(i);
        ymat = Y_MATLAB(i);
        xuni = X_UNITY(i);
        yuni = Y_UNITY(i);

        MATLAB(2*i-1,:) = [xmat, ymat, 1, 0, 0, 0];
        UNITY(2*i-1) = xuni;
        MATLAB(2*i,:) = [0, 0, 0, xmat, ymat, 1];
        UNITY(2*i) = yuni;
    end

    theta = MATLAB\UNITY;
    afflineParams = [theta(1) theta(2) theta(3); theta(4) theta(5) theta(6)];
end

function moveAudio(tcpClient)
    up = KbName('UpArrow'); down = KbName('DownArrow');
    left = KbName('LeftArrow'); right = KbName('RightArrow');
    enter = KbName('Escape');

    % Takes note of the previous command, if the command is '0' (subject not moving audio), we will not spam send '0' to unity to prevent heavy TCP overhead
    preCmd = '0';
    while true
        % Check for user input
        [~, ~, keyCode] = KbCheck;
        if keyCode(enter)
            break;
        end

        curCmd = '0';
        % If the user pressed 1 button (move one direction)
        if sum(keyCode) == 1
            if keyCode(up)
                curCmd = 'W';
            elseif keyCode(down)
                curCmd = 'S';
            elseif keyCode(left)
                curCmd = 'A';
            elseif keyCode(right)
                curCmd = 'D';
            end
        
        % If the user pressed 2 buttons (move diagonally)
        elseif sum(keyCode) == 2
            if keyCode(up) && keyCode(right)
                curCmd = 'WD';
            elseif keyCode(down) && keyCode(right)
                curCmd = 'SD';
            elseif keyCode(up) && keyCode(left)
                curCmd = 'WA';
            elseif keyCode(down) && keyCode(left)
                curCmd = 'SA';
            end
        end

        % If the user not pressing anything
        if strcmp(curCmd, '0')
            % If the previous command is not '0', we send '0' to unity to stop moving the audio source
            if ~strcmp(preCmd, '0')
                jsonStruct = struct();
                jsonStruct.cat = 'move';
                jsonStruct.data = '0';
                MatlabToUnity(tcpClient, jsonStruct);
            end
        else
            % Send the command to unity
            jsonStruct = struct();
            jsonStruct.cat = 'move';
            jsonStruct.data = curCmd;
            MatlabToUnity(tcpClient, jsonStruct);
            preCmd = curCmd;
        end
        WaitSecs(0.01);
    end
end

function audioCoords = retrieveCoords(tcpClient)
    jsonStruct = struct();
    jsonStruct.cat = 'get';

    audioCoords = MatlabUntiyDualCommunication(tcpClient, jsonStruct);
end

function endAudio(tcpClient)
    jsonStruct = struct();
    jsonStruct.cat = 'end';

    MatlabToUnity(tcpClient, jsonStruct);
end

function timeCollection = startAudio(tcpClient, startBool, coordinates, audioPath, timeCollection)
    dataStructStart = struct();
    dataStructStart.start = startBool;
    dataStructStart.coords = coordinates;
    dataStructStart.audio = audioPath;
    jsonStart = jsonencode(dataStructStart);

    jsonStruct = struct();
    jsonStruct.cat = 'start';
    jsonStruct.data = jsonStart;

    jsonStart = jsonencode(jsonStruct);
    jsonStart = [jsonStart, newline];

    % This section measures the round trip time from sending this message to recieving a reply once we start the audio (last tested time: 13 ms)
    startTime = posixtime(datetime('now', 'TimeZone', 'UTC')) * 1000;
    write(tcpClient, uint8(jsonStart));

    % Wait for unity to send back a reply (tells us that the audio has started playing) => Done to measure round trip time
    while true
        if tcpClient.NumBytesAvailable > 0
            endTime = posixtime(datetime('now', 'TimeZone', 'UTC')) * 1000;
            data = read(tcpClient, tcpClient.NumBytesAvailable, 'uint8');
            break;
        end
    end
    
    timeDiff = (endTime - startTime);
    timeCollection = [timeCollection, timeDiff];
end

function MatlabToUnity(tcpClient, jsonStruct)
    jsonStart = jsonencode(jsonStruct);
    jsonStart = [jsonStart, newline];

    write(tcpClient, uint8(jsonStart));
end

function audioCoords = MatlabUntiyDualCommunication(tcpClient, jsonStruct)
    jsonStart = jsonencode(jsonStruct);
    jsonStart = [jsonStart, newline];

    write(tcpClient, uint8(jsonStart));

    while true
        if tcpClient.NumBytesAvailable > 0
            coordinateData = read(tcpClient, tcpClient.NumBytesAvailable, 'uint8');
            jsonString = char(coordinateData);
            jsonObj = jsondecode(jsonString);
            audioCoords = [jsonObj.x, jsonObj.y, jsonObj.z];
            break;
        end
        pause(0.1);
    end
end

function translatedCoords = computeTranslatedCoordinates(coords, theta)
    if size(coords, 2) == 1
        coords = coords.';
    end

    coordsXY = coords(:,1:2);

    TRANSFORM = [theta; 0 0 1];
    n = size(coordsXY, 1);

    newDimCoords = [coordsXY, ones(n, 1)];
    transformedCoords = (TRANSFORM * newDimCoords')';
    translatedXY = transformedCoords(:,1:2);

    z = coords(:,3);
    translatedCoords = [translatedXY, z];
end

function param = audioProcess(tcpClient, param, t)
    iteration = 1;
    stop = 0;

    % If its the first time doing calibration, perform calibration first then validation and repeat process again if the accuracy is not high enough
    if t==1
        while stop==0
            param = audioCalibration(tcpClient, param, iteration);
            stop = audioValidation(tcpClient, param, iteration, stop);
            iteration = iteration + 1;
        end
    
    % If audio calibration triggered halfway (caused by re-conducting the eye calibration), perform validation first. If accuracy is not high enough, redo calibration and perform validation again
    else
        stop = audioValidation(tcpClient, param, iteration, stop);
        while stop==0
            iteration = iteration + 1;
            param = audioCalibration(tcpClient, param, iteration);
            stop = audioValidation(tcpClient, param, iteration, stop);
        end
    end
end

function param = audioCalibration(tcpClient, param, iteration)
    Screen('FillRect', param.window, param.gray);
    line1 = sprintf(['Starting Audio Calibration:\n' ...
        'Use the ARROW KEYS to move the sound until it reaches the white circle'], iteration);
    DrawFormattedText(param.window, line1, 'center', 'center', param.black);
    Screen('flip', param.window);
    WaitSecs(param.AUDIOPRESENTATIONTIME);

    % Circle locations have the row and column of each of the 6 quadrants (we select the 6 quadrants in a random order)
    circleLocationsCalib = param.circleLocationsCalib;
    randomizedLocations = circleLocationsCalib(randperm(size(circleLocationsCalib, 1)), :);

    % We save the coordinates to create the affine transformation later
    X_MATLAB = zeros(9, 1);
    Y_MATLAB = zeros(9, 1);
    X_UNITY = zeros(9, 1);
    Y_UNITY = zeros(9, 1);

    for i=1:size(randomizedLocations, 1)
        coords = randomizedLocations(i, :);
        row = coords(1); col = coords(2);

        % Get the centre of the quadrant we are analysing
        cx = param.quad_width*(col-1) + param.quad_width/2;
        cy = param.quad_height*(row-1) + param.quad_height/2;

        % Drawing the white circle on the black screen
        Screen('FillRect', param.window, [0 0 0]);
        circleRect = [cx - 50, cy - 50, cx + 50, cy + 50];
        Screen('FillOval', param.window, [255 255 255], circleRect);

        Screen('fillrect', param.window, param.white, [param.ctrx-param.half_fix_size param.ctry-param.half_fix_width param.ctrx+param.half_fix_size param.ctry+param.half_fix_width]);
        Screen('fillrect', param.window, param.white, [param.ctrx-param.half_fix_width param.ctry-param.half_fix_size param.ctrx+param.half_fix_width param.ctry+param.half_fix_size]);

        Screen('Flip', param.window);

        % Calculate the real life distance from the centre of the image to the target (based on the lab monitor size) => Maps to unity coordinates (1 unity unit = 1 meter in real life)
        cx = cx-(1920/2);
        cy = -(cy-(1080/2));
        cx = (cx/(1920/2)) * param.monitor_half_width;
        cy = (cy/(1080/2)) * param.monitor_half_height;

        % Set 0m as the depth for calibration (can always edit depending on preference) => Setting too low of a value will cause audio to jump really fast when users are using arrow keys to move audio
        cz = param.monitor_depth + 0;

        % Send command to unity to start playing the audio => Change audio path accordingly
        startAudio(tcpClient, true, {[cx, cy, cz]}, {[fullfile(fileparts(mfilename('fullpath')), 'calib', filesep, '1_untrimmed.wav')]}, []);

        % Starts the function which allows subject to move the audio object in unity to reach the white circle
        moveAudio(tcpClient);

        % Get final coordinates from unity
        audioCoords = retrieveCoords(tcpClient);

        % Stop the audio playing on unity
        endAudio(tcpClient);

        % Save the current and moved coordinates
        X_MATLAB(i) = cx;
        Y_MATLAB(i) = cy;

        X_UNITY(i) = audioCoords(1);
        Y_UNITY(i) = audioCoords(2);
        WaitSecs(0.2);
    end

    % Compute the parameters of the affine transformation and saves them
    theta = computeAfflineParameters(X_MATLAB, Y_MATLAB, X_UNITY, Y_UNITY);
    param.theta = theta;
end

function stop = audioValidation(tcpClient, param, iteration, stop)
    window = param.window;
    window_rect = param.window_rect;
    circleLocationsCalib = param.circleLocationsCalib;
    circleLocationsCalibValid = param.circleLocationsCalibValid;

    Screen('FillRect', window, param.gray);
    line1 = sprintf(['Starting Audio Validation:\n' ...
        'Press the quadrant where the sound comes from'], iteration);
    DrawFormattedText(window, line1, 'center', 'center', param.black);
    Screen('flip', window);
    WaitSecs(param.AUDIOPRESENTATIONTIME);

    % Create a black screen with the 2 quadrants highlighted
    stimuli_img = zeros(window_rect(4), window_rect(3), 2, 'uint8');
    for col=1:1
        x = round(col*param.window_rect(3)/2);
        stimuli_img(:,x,:) = 255;
    end

    textures = Screen('MakeTexture', window, stimuli_img);
    Screen('DrawTexture', window, textures, window_rect);
    Screen('Flip', window);

    % Randomly select a quadrant
    randomIndices = randperm(size(circleLocationsCalib, 1));
    randomizedLocations = circleLocationsCalib(randomIndices, :);
    randomizedQuadrants = circleLocationsCalibValid(randomIndices, :);

    count = 0;
    for i=1:size(randomizedLocations, 1)
        coords = randomizedLocations(i, :);
        row = coords(1); col = coords(2);

        coordsQuad = randomizedQuadrants(i, :);
        curQuadrant = coordsQuad(1) + coordsQuad(2) - 1;

        % Same coordinate calculation as the audio calibration
        cx = param.quad_width*(col-1) + param.quad_width/2;
        cy = param.quad_height*(row-1) + param.quad_height/2;

        cx = cx-(1920/2);
        cy = -(cy-(1080/2));

        cx = (cx/(1920/2)) * param.monitor_half_width;
        cy = (cy/(1080/2)) * param.monitor_half_height;

        cz = param.monitor_depth + 0;

        % Compute the final audio to be placed in unity using the affine transformation
        translatedCoords = computeTranslatedCoordinates([cx, cy, cz], param.theta);
        inputCoords = translatedCoords(1, :);

        % Start playing the audio in unity => Change path when needed
        startAudio(tcpClient, true, {[inputCoords(1), inputCoords(2), inputCoords(3)]}, {[fullfile(fileparts(mfilename('fullpath')), 'calib', filesep, '1_untrimmed.wav')]}, []);

        ShowCursor;

        % Calculate which quadrant the user actually clicks
        while true
            [x, y, buttons] = GetMouse(window);

            x = int32(x);
            y = int32(y);

            if x > window_rect(3)
                x = window_rect(3);
            end

            if x < 1
                x = 1;
            end

            if y > window_rect(4)
                y = window_rect(4);
            end

            if y < 1
                y = 1;
            end

            if buttons(1) == 1
                endAudio(tcpClient);
                Beeper('high', 0.5, 0.2);
                x = double(x); y = double(y);
                quad_height = double(param.window_rect(4)); quad_width = double(param.window_rect(3)/2);
                clickedRow = floor(y/quad_height) + 1;
                clickedCol = floor(x/quad_width) + 1;
                clickedQuadrant = (clickedCol + clickedRow) - 1;

                display(curQuadrant);
                display(clickedQuadrant);
                
                % If the user pressed the right quadrant, we add 1 to the score
                if clickedQuadrant == curQuadrant
                    count = count + 1;
                end
                break;
            end
        end
    end

    fprintf('****************************************************************\n');
    fprintf('%d out of 6 points identified\n', count);
    fprintf('****************************************************************\n');

    if (count>=5)
        stop = 1;
    end
end
