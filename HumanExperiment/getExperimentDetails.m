function [result] = getExperimentDetails()
txt=sprintf('\n\n\n');disp(txt);
valid_data=0;
while (valid_data==0)
    prompt ={'Please enter the subject ID (e.g. subj05-basu)', ...
        'Age:', 'Glasses', ...
        'Eyelink:', 'Experiment Mode (0 exp; 1 test):', 'Session:', 'Starting block:'};
    
    valid_name = 0;
    %answercell = inputDialog(prompt, 'INPUT', 1, ...
        %{'subj', '', '0', '1', num2str((rand() > 0.5) + 1), '1', '1'});
    answercell = inputdlg(prompt, 'INPUT', 1, ...
        {'subj01-xu', '25', '1', '0', '0', '1', '1'});
    if isempty(answercell)
        error('cancelled by user');
    end
    subject_initials=answercell{1};
    if (ischar(subject_initials))
        aux=findstr(subject_initials,' ');
        if (~isempty(aux))
            txt=sprintf('No spaces allowed');disp(txt);
        else
            aux=findstr(subject_initials,'_');
            if (~isempty(aux))
                txt=sprintf('No ''_'' allowed');disp(txt);
            else
                ll=length(subject_initials);
                if ( (ll<2) || (ll>20) )
                    txt=sprintf('length=%d -- must be between 2 and 20',ll);disp(txt);
                else
                    valid_name=1;
                end
            end
        end
    end
    
    validOcclusionType = 0;
    occlusion_type = str2num(answercell{5});
    if(ismember(occlusion_type,[0 1]))
        validOcclusionType = 1;
    else
        fprintf(['Invalid occlusion type ' ...
            '(must be either 0 for experiment or 1 for test mode).\n']);
    end
    
    validSession = 0;
    session = str2num(answercell{6});
    if session > 0 && session <= 4
        validSession = 1;
    else
        fprintf('Invalid session number.\n');
    end
    
    valid_data = valid_name & validOcclusionType & validSession;
    if ~valid_data
        error('Invalid data');
    end
    
end

%% search the subjects directory for the subject number

if(~strcmp(subject_initials,'subj01-test'))    
    idPattern = 'subj(\d+)';
    subjectId = regexp(subject_initials, idPattern, 'tokens');
    
    if(isempty(subjectId))
        error('**subject name must be of right format: subjXX-Lastname (e.g. subj01-test)');
    else
        subjectId = str2num(subjectId{1}{1});
    end
    
    subjectsDir = [fileparts(mfilename('fullpath')) '/../subjects'];
    if ~exist(subjectsDir, 'dir')
        mkdir(subjectsDir);
    end
    D = dir([subjectsDir '/subj*']);
    
    % does the folder already exist?
    folder_exists = 0;
    for i = 1:length(D)
        if(D(i).isdir)
            try
                otherSubjectId = regexp(D(i).name,idPattern,'tokens');
                otherSubjectId = str2num(otherSubjectId{1}{1});
                
                % check if folder already exists
                if(otherSubjectId == subjectId)
                    if(~strcmp(D(i).name,subject_initials))
                        error('**You entered %s, but folder already exists named %s!',subject_initials,D(i).name);
                    else
                        folder_exists = 1;
                    end
                end
            catch
            end
        end
    end
    
    % if folder is new, the session number should = 1
    if(folder_exists == 0)
        if(session ~= 1)
            error('**New subject, so session number should be 1');
        end
    end
    
    % if folder is old, the session number should increment
    if(folder_exists == 1)
        D = dir(sprintf('%s/%s/*.mat', subjectsDir, subject_initials));
        if(isempty(D) && session ~= 1)
            error('**New subject, so session number should be 1');
        else
            
            latestSession = 0;
            for i = 1:length(D)                
                sessionPattern = 'sess(\d+)_';
                sess = regexp(D(i).name,sessionPattern,'tokens');
                if isempty(sess)
                    % different format was used earlier. ignore
                    continue;
                end
                sess = str2double(sess{1}{1});
                latestSession = max(latestSession,sess);
            end
            
            if latestSession > 0 && session ~= latestSession+1
                fprintf('** WARNING: Previous session was session number %d, so todays session should be %d.\n',latestSession,latestSession+1);
                fprintf('** You entered session %d.\n',session);
                answer = input('** This is not correct unless in certain situations! Continue? [y/n]$ ','s');
                if ~strcmpi(answer,'y')
                    error('Please rerun the experiment with correct data.');
                end
                
            end
        end
    end
    
end
%%


result.subject_initials = subject_initials;
result.subject_age = str2num(answercell{2});
result.subject_glasses = str2num(answercell{3});
result.valid_data = valid_data;
result.eyelink = str2num(answercell{4});
result.ExperimentMode = occlusion_type;
result.session = session;
result.block2start = str2num(answercell{7});
