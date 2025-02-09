function audioValidation(window, windowRect)
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
    
    correctCount = 0;
    while true
        randomOrder = randperm(length(audioFiles));
        DrawFormattedText(window, 'audio validation and enter the space to begin', 'center', screenYpixels * 0.45, BlackIndex(window));
        Screen('Flip', window);
        KbWait;
        
        for i = 1:length(randomOrder)
            Screen('FillRect', window, [128 128 128]);
            HideCursor;
            for x = 2:length(xGrid)-1
                Screen('DrawLine', window, [0 0 0], xGrid(x), 0, xGrid(x), screenYpixels, 2);
            end
            for y = 2:length(yGrid)-1
                Screen('DrawLine', window, [0 0 0], 0, yGrid(y), screenXpixels, yGrid(y), 2);
            end
            Screen('Flip', window);
            
            audioFile = audioFiles{randomOrder(i)};
            [y, Fs] = audioread(audioFile);
            player = audioplayer(y, Fs);
            playblocking(player);
            ShowCursor;
            for x = 2:length(xGrid)-1
                Screen('DrawLine', window, [0 0 0], xGrid(x), 0, xGrid(x), screenYpixels, 2);
            end
            for y = 2:length(yGrid)-1
                Screen('DrawLine', window, [0 0 0], 0, yGrid(y), screenXpixels, yGrid(y), 2);
            end
            Screen('Flip', window);
            
            clicked = false;
            startTime = GetSecs();
            while GetSecs() - startTime < 5
                [x, y, buttons] = GetMouse(window);
                if any(buttons)
                    clicked = true;
                    % 检查点击位置是否正确
                    xPos = find(xGrid <= x, 1, 'last');
                    yPos = find(yGrid <= y, 1, 'last');
                    if isequal([xPos, yPos], correctPositions(i, :))
                        correctCount = correctCount + 1;
                    end
                    break;
                end
            end
        end
        
        if correctCount >= 1
            DrawFormattedText(window, 'success validation,  enter the space to begin the experiment!', 'center', screenYpixels * 0.45, BlackIndex(window));
            Screen('Flip', window);
            KbWait;
            break;
        else
            DrawFormattedText(window, 'failure validation, you need to validate again.', 'center', screenYpixels * 0.45, BlackIndex(window));
        end
        Screen('Flip', window);
        WaitSecs(1);
        KbWait;
    end
end