%% Lee et al., Nature 2016 Fig. 4b
% Are bigger synapses between functionally similar cells?

load('Synapse_Info.mat')
load('connSyns.mat')

dOri = abs(connSyns(:,7) - connSyns(:,11));
dOri(abs(dOri)>90) = 180.0 - abs(dOri(abs(dOri)>90));

[C,iA,iB] = intersect(connSyns(:,1),Synapse_Info(:,1));
dOri = dOri(iA);
psdVol = Synapse_Info(iB,5); % 1: synID, 2: voxels, 3: voxels interp, 4: nm^2, 5: nm^2 interp

%% Bar jitter
numBars = 4;

%
if numBars == 3
    bins = [0,22.5,67.5,90];
elseif numBars == 4
    bins = [0,22.5,45,67.5,90];
elseif numBars == 2
    bins = [0,45,90];
end

[bincounts,ind] = histc(dOri,bins);
[Ymean,Ysem,Yci,Yn] = grpstats(psdVol,ind,{'mean','sem','meanci','numel'});

figure
hold on
plot(ind,psdVol,'ok','MarkerSize',7)
c=get(gca,'Children'); % The second handle is that for the first plotted data
x=get(c(1),'XData'); % Retrieve the X data for the black points

%Add uniformly distributed random numbers to the data and re-plot
r=rand(size(x))/5;
r=r-mean(r);
set(c(1),'XData',x+r)

title('\Delta Ori vs. Synaptic (PSD) Area');
xlabel('\Delta Orientation (degrees)');
ylabel('Synaptic (PSD) Area (\mum^2)');

if numBars == 3
%     axis([.5 3.5 0 1.2*(max(Ymean)+max(Ysem))])
    set(gca,'XTick',1:3,'XTickLabel',{'0','45','90'})
elseif numBars == 4
%     axis([.5 4.5 0 1.2*(max(Ymean)+max(Ysem))])
    set(gca,'XTick',1:4,'XTickLabel',{'0','22.5-45','45-67.5','90'})
elseif numBars == 2
%     axis([.5 2.5 0 1.2*(max(Ymean)+max(Ysem))])
    xlim([.5 2.5])
    set(gca,'XTick',1:2,'XTickLabel',{'< 45','> 45'})       
end

yL = get(gca,'yLim');

bar(Ymean,.4,'FaceColor','none','LineWidth',1)
errorbar(Ymean,Ymean-Yci(:,1),'k','LineStyle','none','LineWidth',1)
ylim([0 0.375])