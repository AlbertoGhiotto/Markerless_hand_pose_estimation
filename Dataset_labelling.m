clc;
clear all;
close all;

N = 315;    % # of images to be labelled
f = 16;    % # of features
ex = imread('../group_project/HowToLabel/HowToLabel.jpg'); 

disp('Choose the keypoints(joints) as shown in the example image');
figure, imshow(ex), title('Choose the keypoints(joints) as shown in the example image,')
(' if the feature is not visible select a point over the edges of the image');
labels = zeros(N,f,2);   % tridimensional matrix for label coordinates

k = 1;
while k <= N
    I = imread(['../group_project/dataset/img (' , num2str(k) , ').jpg']);  %change name
    [len_y, len_x, channels] = size(I);  % get the dimension of the images

    figure, imshow(I), title(['Image #', num2str(k)]), hold on;
    for n = 1 : f
        labels(k,n,:) = ginput(1);
        plot(labels(k,n,1), labels(k,n,2), 'sr');
        text(labels(k,n,1)-20, labels(k,n,2)-20, sprintf('%d', n));
        if (labels(k,n,1) < 0 || labels(k,n,1) > len_x  || labels(k,n,2) < 0 || labels(k,n,2) > len_y  )
        labels(k,n,:) = [-1,-1];            % if the feature is put over the edges of the image 
                                            % -> put -1 to consider the
                                            % feature as nonexistent
        end
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
    
    