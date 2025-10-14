subj_dir = 'Subjects';
subject = 'subj04-yh';

full_dir = [fullfile(fileparts(mfilename('fullpath')), subj_dir, subject, filesep)];
save_path = [subject '.mat'];
mat_files = dir(fullfile(full_dir, '*.mat'));

numTrials = 90;
t = 1;
Fix_posx = {};
Fix_posy = {};
target_found = [];
num_fixations = [];
FixData = [];

for i=1:length(mat_files)
    data_path = fullfile(mat_files(i).folder, mat_files(i).name);
    display(data_path);
    data = load(data_path).eventlist;

    searchString = ['ENDFIX'];
    fixindex = findString(data,'codestring',searchString);

    searchString = ['TARGET FOUND'];
    targetFound = findString(data, 'message', searchString);
    
    searchString = ['TIME EXCEED'];
    targetNotFound = findString(data, 'message', searchString);

    display(length(targetFound));
    display(length(targetNotFound));
    
    while t<=numTrials
        searchString = ['TRIAL_ON: ' num2str(t)];
        startindex=findString(data, 'message', searchString);

        searchString = ['TRIAL_OFF: ' num2str(t)];
        endindex=findString(data, 'message', searchString);

        if isempty(startindex) || isempty(endindex)
            break;
        end

        startindex = startindex(1);
        endindex = endindex(1);

        filteredfixindex = fixindex(find(fixindex >= startindex & fixindex <= endindex+1));
        fixx = [];
        fixy = [];
        % fixtime = [];
        % fixstarttime = [];
        
        for fx=1:length(filteredfixindex)
            fixx = [fixx, data(filteredfixindex(fx)).gavx];
            fixy = [fixy, data(filteredfixindex(fx)).gavy];
            
        end
        
        Fix_posx = [Fix_posx; int32(fixx)];
        Fix_posy = [Fix_posy; int32(fixy)];
        target_found = [target_found, ~isempty(find(targetFound >=startindex & targetFound <= endindex))];
        num_fixations = [num_fixations, length(fixx)];

        t = t+1;
    end
end

FixData.Fix_posx = Fix_posx;
FixData.Fix_posy = Fix_posy;
FixData.TargetFound = target_found;
FixData.Fix_num = num_fixations;
save(save_path,'FixData');

function index = findString(search_list, search_param, str)
    index = [];

    if strcmp(search_param, 'codestring')
        for f=1:length(search_list)    
            compare = search_list(f).codestring;
            if contains(compare, str)
                index = [index, f];
            end
        end
    end
    
    if strcmp(search_param, 'message')
        for f=1:length(search_list)
            if isempty(search_list(f).message)
                continue;
            end
    
            compare = search_list(f).message;
            if contains(compare, str)
                index = [index, f];
            end
        end
    end
end