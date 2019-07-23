clc;
clear all;
close all;

N = 2;    % # of images to be labelled
f = 16;    % # of features
ex = imread('../GP/Images/img_Ex.jpg'); 

% disp('Choose the keypoints(joints) as shown in the example image');
% figure, imshow(ex), title('Example image')
labels = zeros(N,f,2);

k = 1;
while k <= N
    I = imread(['../GP/Images/Hand_000000' , num2str(k) , '.jpg']);  %change name
    figure, imshow(I), title(['Image #', num2str(k)]), hold on;
    for n = 1 : f
        labels(k,n,:) = ginput(1);
        plot(labels(k,n,1), labels(k,n,2), 'sr');
        text(labels(k,n,1)-20, labels(k,n,2)-20, sprintf('%d', n));
             
    end
    prompt = 'Is the labelling correct?(y->1/n->0)';
    choice = input(prompt);
    if (choice == 0)
       
    else
        k = k+1;
    end
end
delim = '-----';
dlmwrite('dataset.csv', delim , 'delimiter' , ',','-append'); 
dlmwrite('dataset.csv', labels, 'delimiter' , ',','-append'); 

% writematrix(,'dataset.csv')
% writematrix(labels,'dataset.csv')
    
    