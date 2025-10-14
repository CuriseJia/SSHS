fileID = fopen('result.txt', 'r');
imageInfo = {};
imagePath = {};
ImageDir = "C:\Users\DeepN\OneDrive\Desktop\yanhao_new\dataset\image\";
lineCount = 0;
while ~feof(fileID)
    line = fgetl(fileID);
    if ischar(line)
        lineCount = lineCount + 1;
        if lineCount > 1
            data = strsplit(line, '\t');
            imageInfo{end+1} = strtrim(data{3});
            imagePath{end+1} = ImageDir + strtrim(data{3});
        end
    end
end
fclose(fileID);

subject = 'subj04-yh';
data = load([fullfile(fileparts(mfilename('fullpath')), [subject '.mat'])]);


for j=1:length(imagePath)
    img = imagePath(j);
    img = img{1};
    stimuli_img_path = img;
    stimuli_img = imread(stimuli_img_path);
    stimuli_img = imresize(stimuli_img, [1080 1920]);

    pre_fixx = data.FixData.Fix_posx(j);
    pre_fixy = data.FixData.Fix_posy(j);
    fixx = pre_fixx{1};
    fixy = pre_fixy{1};

    [imgHeight, imgWidth, ~] = size(stimuli_img);

    centerX = imgWidth / 2;
    centerY = imgHeight / 2;

    offsetx = centerX - fixx(1);
    offsety = centerY - fixy(1);

    fixx = double(fixx) + double(offsetx);
    fixy = double(fixy) + double(offsety);

    fixnumstr = cellstr(string(1:1:length(fixx)));
    RGB = insertText(stimuli_img,[int32(fixx(1:end)); int32(fixy(1:end))]', fixnumstr);

    for i = 1:length(fixx)-1
        RGB = insertShape(RGB, 'Line', [fixx(i), fixy(i), fixx(i+1), fixy(i+1)], 'Color', 'green', 'LineWidth', 2);

        % Calculate the angle of the line
        angle = atan2(fixy(i+1) - fixy(i), fixx(i+1) - fixx(i));
        
        % Define the arrowhead parameters
        arrowLength = 20; 
        arrowWidth = 10;
        
        % Calculate the points of the arrowhead
        arrowX = fixx(i+1) - arrowLength * cos(angle);
        arrowY = fixy(i+1) - arrowLength * sin(angle);
        arrowPoints = [arrowX - arrowWidth * sin(angle), arrowY + arrowWidth * cos(angle), ...
                       fixx(i+1), fixy(i+1), ...
                       arrowX + arrowWidth * sin(angle), arrowY - arrowWidth * cos(angle)];
        
        % Draw the arrowhead
        RGB = insertShape(RGB, 'FilledPolygon', arrowPoints, 'Color', 'green');
    end
    
    check = imageInfo(j);
    check = string(check);
    save_path = string([fullfile(fileparts(mfilename('fullpath')), 'FixationPlot', subject)]);
    display(save_path);
    display(check);
    save_path = strcat(save_path, '\', check);
    display(save_path)
    imwrite(RGB, save_path);
end