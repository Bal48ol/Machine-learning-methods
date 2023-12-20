function word_indices = processEmail(email_contents)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices
%   word_indices = PROCESSEMAIL(email_contents) preprocesses
%   the body of an email and returns a list of indices of the
%   words contained in the email.
%

% Load Vocabulary
vocabList = getVocabList();

% Init return value
word_indices = [];

% ========================== Preprocess Email ===========================

% Find the Headers ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);

% Lower case
email_contents = lower(email_contents);

% Strip all HTML
% Looks for any expression that starts with < and ends with > and replace
% and does not have any < or > in the tag it with a space
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% Handle Numbers
% Look for one or more characters between 0-9
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% Handle URLS
% Look for strings starting with http:// or https://
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% Handle Email Addresses
% Look for strings with @ in the middle
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Handle $ sign
email_contents = regexprep(email_contents, '[$]+', 'dollar');


% ========================== Tokenize Email ===========================

% Output the email to screen as well
fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;

while ~isempty(email_contents)

    % Tokenize and also get rid of any punctuation
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);

    % Remove any non alphanumeric characters
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str));
    catch str = ''; continue;
    end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end

    % Найдите слово в словаре и добавьте его позицию в список word_indices
    % ====================== НАЧАЛО ВАШЕГО КОДА ======================
    % Инструкции: К текущему моменту у вас есть предобработанное слово в переменной str.
    %               Вы должны выполнить поиск этого слова в словаре vocabList.
    %               Имейте в виду, что под словарем здесь понимается обычный список слов,
    %               а не стуктура данных "ассоциативный массив".
    %               Если слово найдено, то в список word_indices надо добавить позицию этого слова в словаре.
    %               Например, если str = 'action' и оно есть в словаре:
    %               vocabList{18} = 'action', то вам надо добавить число 18 в word_indices:
    %               word_indices = [word_indices ; 18];
    %
    % Замечание: vocabList{idx} возвращает слово с индексом idx в словаре
    %
    % Замечание: Вы можете использовать strcmp(str1, str2) для сравнения двух строк.
    %            Функция возвращает 1, если строки равны.
    %


    % Находим слово в словаре и добавляем его позицию в список word_indices
    for i = 1:length(vocabList)
      if strcmp(str, vocabList{i})
        word_indices = [word_indices ; i];
        break;
      end
    end

    % ====================== КОНЕЦ ВАШЕГО КОДА ======================


    % Print to screen, ensuring that the output lines are not too long
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;

end

% Print footer
fprintf('\n\n=========================\n');

end
