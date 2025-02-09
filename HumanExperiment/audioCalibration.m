function audioCalibration(window, windowRect)
    audioFiles = {'audio/af_1.flac.wav', 'audio/af_3.flac.wav', ...
                  'audio/af_7.flac.wav', 'audio/af_9.flac.wav'};
    correctPositions = [
        1, 1;
        1, 2;
        2, 1;
        2, 2;
    ];
    [screenXpixels, screenYpixels] = Screen('WindowSize', window);
    xGrid = linspace(0, screenXpixels, 3);
    yGrid = linspace(0, screenYpixels, 3);
    line1 = 'Ready.';
    line2 = '\n Press ENTER key to begin the audio calibration process.';
    line3 = '\n In this process, you will hear a series of audio clips and their corresponding regions will be displayed on the screen.';
    line4 = '\n Make sure that you can hear the audio clips clearly and press Enter key to continue the audio validation process.';
    DrawFormattedText(window, [line1 line2 line3 line4],'center', screenYpixels * 0.45, BlackIndex(window));   
    Screen('Flip',window);    
    WaitSecs(0.2); 
    KbWait();
    while true
        randomOrder = randperm(length(audioFiles));
        Screen('FillRect', window, [128 128 128]);
        HideCursor;
        for x = 2:length(xGrid)-1
            Screen('DrawLine', window, [0 0 0], xGrid(x), 0, xGrid(x), screenYpixels, 2);
        end
        for y = 2:length(yGrid)-1
            Screen('DrawLine', window, [0 0 0], 0, yGrid(y), screenXpixels, yGrid(y), 2);
        end
        Screen('Flip', window);    
        WaitSecs(1);
        
        for i = 1:length(randomOrder)
            % 确定当前音频对应的区域
            currentPos = correctPositions(randomOrder(i), :);
            xPos = currentPos(1);
            yPos = currentPos(2);
            Screen('FrameRect', window, [255 0 0], [xGrid(xPos) yGrid(yPos) xGrid(xPos+1) yGrid(yPos+1)], 2);
            for x = 1:length(xGrid)-1
                for y = 1:length(yGrid)-1
                    if ~(x == xPos && y == yPos)
                        Screen('FrameRect', window, [0 0 0], [xGrid(x) yGrid(y) xGrid(x+1) yGrid(y+1)], 1);
                    end
                end
            end
            
            Screen('Flip', window);
            audioFile = audioFiles{randomOrder(i)};
            [y, Fs] = audioread(audioFile);
            player = audioplayer(y, Fs);
            playblocking(player);
        end
        
        while true
            line1 = 'If you think you can locate the audio clips with corresponding regions successfully, press ENTER key to begin the next process.';
            line2 = '\n Or press SPACE key to re-calibrate the audio clips.';
            DrawFormattedText(window, [line1 line2],'center', screenYpixels * 0.45, BlackIndex(window));   
            Screen('Flip',window);
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown
                if keyCode(KbName('space'))
                    % 重新进行校准操作
                    break;
                elseif keyCode(KbName('Return'))
                    % 结束操作
                    Screen('Flip',window);    
                    WaitSecs(0.2);
                    % Screen('CloseAll');
                    return;
                end
            end
        end
    end
end