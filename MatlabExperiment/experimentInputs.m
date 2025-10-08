function inputs = experimentInputs()

    while true

        % Create dialogue box for experiment
        prompt = {'Subject Name (e.g. subj05-basu):', ...
            'Age:', ...
            'Glasses (0 = No, 1 = Yes):', ...
            'Eyelink (0 = No, 1 = Yes):'};

        default_input = {'subj01-sj', '24', '1', '1'};
        title = 'Experiment Input';

        user_input = inputdlg(prompt, title, 1, default_input);

        if isempty(user_input)
            error("User has cancelled the experiment")
        end
        
        % Obtain user inputs
        subject_name = user_input{1};
        subject_age = str2double(user_input{2});
        glasses = str2double(user_input{3});
        eyelink = str2double(user_input{4});

        % Verify inputs
        if isnan(glasses) || (glasses~=0 && glasses~=1)
            fprintf("Invalid Glasses input, key in either 0 or 1");
            continue;
        end

        if isnan(eyelink) || (eyelink~=0 && eyelink~=1)
            fprintf("Invalid Eyelink input, key in either 0 or 1");
            continue;
        end

        if isnan(subject_age) || (subject_age < 0 || subject_age > 100)
            fprintf("Invalid age input, please try again");
            continue;
        end

        % Verify the subject name
        if contains(subject_name, ' ') || contains(subject_name, '_')
            fprintf("Subject name cannot contain spaces or underscores, please try again");
            continue;
        end

        % Obtain subject id
        regex_statement = 'subj(\d+)';
        subject_id = regexp(subject_name, regex_statement, 'tokens');
        subject_id = subject_id{1};

        display(subject_id);
        if isempty(subject_id)
            fprintf("Format of subject name is wrong (does not include subject id), please try again"); 
            continue;
        end

        subject_id = str2double(subject_id);

        save_path = [fileparts(mfilename('fullpath')) '/../Subjects']; % Maybe check if folder exists
        existing_subj = dir([save_path 'subj*']);

        continue_flag = 0;
        for f=1:length(existing_subj)
            % Check if subject name already exisits
            if strcmp(existing_subj(f).name, subject_name)
                fprintd("Subject name already exists, please try again")
                continue_flag = 1;
                break;
            end
            
            % Check if subject Id already exisits
            file_id = regexp(existing_subj(f).name, regex_statement, 'tokens');

            if str2double(file_id) == subject_id
                fprintf("Subject Id already exisits, please try again")
                continue_flag = 1;
                break;
            end
        end

        if continue_flag==1
            continue;
        end
        break;

    end
    display(subject_id)
    inputs.subjectName = subject_name;
    inputs.subjectId = subject_id;
    inputs.subjectAge = subject_age;
    inputs.subjectGlasses = glasses;
    inputs.eyelink = eyelink;

end