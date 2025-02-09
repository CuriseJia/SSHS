function DrawCenteredText(window, text, color)

% text has to be a cell array of lines

if nargin < 3
   color = 0;
end
win_rect=Screen('Rect', window);
total_height=0;
for ln=1:length(text);
   bbox{ln} = Screen('TextBounds',window,text{ln});
   total_height=total_height+bbox{ln}(RectBottom);
   bbox{ln} = CenterRect(bbox{ln},win_rect);
end
y=win_rect(RectTop)+(win_rect(RectBottom)-win_rect(RectTop)-total_height)/2;
for ln=1:length(text);
   x=bbox{ln}(RectLeft);
   Screen('DrawText',window,text{ln},x,y,color);
   y=y+bbox{ln}(RectBottom)-bbox{ln}(RectTop);
end