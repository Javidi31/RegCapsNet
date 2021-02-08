
%%%%%%%%%%%%%%%%%%%
% Authors: Mahdi Jampour. Email: jampour [at] icg.tugraz.at
% Authors: Malihe Javidi. Email: m.javidi [at] qiet.ac.ir



function Preparing
close all; clear; clc;
MyPath1 = '\dataset\CedarOrg\';    
MyPath2 = '\dataset\Cedar\';    
    % Download CEDAR dataset and run CorrectFileNames only one time to
    % Correct fileneames
    % CorrectFileNames(MyPath);
    
    Files  = dir([MyPath1 '\*.png']);
    for j=1:length(Files)
        fn1 = [MyPath1 Files(j,1).name];
        fn2 = [MyPath2 Files(j,1).name];
        im1 = imread(fn1);
        [r,c,z]=size(im1);
        disp(fn1)
        if(z>1), im1 = rgb2gray(im1); end       
        %im2 = IgnoreBesideSpace(im1);
        im2 = uint8(255*im2bw(im1, 0.75));
        im3 = Change2SqureImage(im2);
        im4 = imresize(im3, 0.5);
        imwrite(im4, fn2);
    end
end


function CorrectFileNames(MyPath)
    % Correct filenames from 1 to 01, 2 to 02, etc.
    Files  = dir([MyPath '\*.png']);
    for j=1:length(Files)
        fn1 = [MyPath Files(j,1).name];
        fn2 = fn1;
        if(    strfind(Files(j,1).name, '_1_')>0)
            fn2 = strrep(fn1, '_1_',    '_01_');
        elseif(strfind(Files(j,1).name, '_2_')>0)
            fn2 = strrep(fn1, '_2_',    '_02_');
        elseif(strfind(Files(j,1).name, '_3_')>0)
            fn2 = strrep(fn1, '_3_',    '_03_');
        elseif(strfind(Files(j,1).name, '_4_')>0)
            fn2 = strrep(fn1, '_4_',    '_04_');
        elseif(strfind(Files(j,1).name, '_5_')>0)
            fn2 = strrep(fn1, '_5_',    '_05_');
        elseif(strfind(Files(j,1).name, '_6_')>0)
            fn2 = strrep(fn1, '_6_',    '_06_');
        elseif(strfind(Files(j,1).name, '_7_')>0)
            fn2 = strrep(fn1, '_7_',    '_07_');
        elseif(strfind(Files(j,1).name, '_8_')>0)
            fn2 = strrep(fn1, '_8_',    '_08_');
        elseif(strfind(Files(j,1).name, '_9_')>0)
            fn2 = strrep(fn1, '_9_',    '_09_');
        end
        fprintf('%s -> %s \n', fn1, fn2);
        
        if(length(fn1)~=length(fn2))
            movefile(fn1, fn2);
        end
    end
    
    for j=1:length(Files)
        fn1 = [MyPath Files(j,1).name];
        fn2 = fn1;
        if(    strfind(Files(j,1).name, '_1.')>0)
            fn2 = strrep(fn1, '_1.',    '_01.');
        elseif(strfind(Files(j,1).name, '_2.')>0)
            fn2 = strrep(fn1, '_2.',    '_02.');
        elseif(strfind(Files(j,1).name, '_3.')>0)
            fn2 = strrep(fn1, '_3.',    '_03.');
        elseif(strfind(Files(j,1).name, '_4.')>0)
            fn2 = strrep(fn1, '_4.',    '_04.');
        elseif(strfind(Files(j,1).name, '_5.')>0)
            fn2 = strrep(fn1, '_5.',    '_05.');
        elseif(strfind(Files(j,1).name, '_6.')>0)
            fn2 = strrep(fn1, '_6.',    '_06.');
        elseif(strfind(Files(j,1).name, '_7.')>0)
            fn2 = strrep(fn1, '_7.',    '_07.');
        elseif(strfind(Files(j,1).name, '_8.')>0)
            fn2 = strrep(fn1, '_8.',    '_08.');
        elseif(strfind(Files(j,1).name, '_9.')>0)
            fn2 = strrep(fn1, '_9.',    '_09.');
        end
        fprintf('%s => %s \n', fn1, fn2);
        
        if(length(fn1)~=length(fn2))
            movefile(fn1, fn2);
        end
    end    
end

function pic = IgnoreBesideSpace(img)
    Xax = sum(img);
    Yax = sum(img, 2);

    %==Detect X=============================================
    MaximX = max(Xax);
    for k=1:size(img, 2)
        if(Xax(1,k)<MaximX)
            x1 = k;
            break;
        end
    end
    for k=size(img, 2):-1:1
        if(Xax(1,k)<MaximX)
            x2 = k;
            break;
        end
    end        
    %==Detect Y=============================================
    MaximY = max(Yax);
    for k=1:size(img, 1)
        if(Yax(k,1)<MaximY)
            y1 = k;
            break;
        end
    end
    for k=size(img, 1):-1:1
        if(Yax(k,1)<MaximY)
            y2 = k;
            break;
        end
    end        
    %=======================================================
    pic = img(y1:y2, x1:x2);
end

function pic = Change2SqureImage(img)
    [x,y]=size(img);
    z = max(x,y)+20;
    pic = uint8(255*ones(z,z));
    
    xx = floor((z-x)/2);
    yy = floor((z-y)/2);
    pic(xx:x+xx-1, yy:y+yy-1) = img;
    pic = imresize(pic, [400 400]);
end

function pic = Rotate18SigImage(img)
    [x,y] = size(img);
    for i=1:x
        for j=1:y
            if(img(i,j)<2)
                img(i,j)=2;
            end
        end
    end    

    im1 = imbinarize(img);
    for k=1:18
        im2 = imrotate(img, k*20, 'crop');
        for i=1:x
            for j=1:y
                if(im2(i,j)==0)
                    im2(i,j)=255;
                end
            end
        end
        im1 = im1 .* imbinarize(im2);
    end
    
    pic = uint8(255*im1);
end
