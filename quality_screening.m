clear
close all
dname = uigetdir();% add default folder if preferred
file=dir(fullfile(dname, '*.mat'));
file = file(contains({file.name}, '_score_and_feature.mat'));
close all
for i=1:length(file)
    p=split(file(i).name,'_');
    load([file(i).folder '\' file(i).name])
    EMG=EMG_features(:,1);
    EMG=[EMG;zeros(21600-length(EMG),1)];
    score1=[score';zeros(21600-length(score'),1)];
    figure(i)
    % plot(DT)    
    plot(EEG_feautures(:,2)+12)
    hold on
    plot(EEG_feautures(:,1)+8)
    plot(EMG+4) 
    stairs(score1) 
    legend('Theta','Delta','EMG','Score')
    wake=length(find(score==1))/length(score)*100;
    NREM=length(find(score==2))/length(score)*100;
    REM=length(find(score==3))/length(score)*100;
    % xlim(lim)
    xlim([0 10000])
    title([p{1} '|' p{2} '|' p{3} '|' p{4} '|' 'wake=' num2str(wake) '%,' 'NREM=' num2str(NREM) '%' 'REM=' num2str(REM) '%'],'interpreter','none','FontSize',12)
end

%note: use the horizontal zoom function in matlab figure page to zoom into
%small portion of the plot for closer examination
%sleep and wake should be around 50/50, 5%of total time should be REM