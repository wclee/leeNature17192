%% Modularity analysis
% the empirical network
load('lee_deg2adjacency_nature17192.mat') % deg >1 adjacency matrix
load('lee_deg2ids_nature17192.mat') % skeleton (neuron) IDs for adjacency

% run mod calculation N_mod times and inspect modularity values
N_mod = 1000
cpus = 'local'; % # nodes for parallelizing

ModMatL=[]; % initialize vector for Louvain computed modularity values
GroupsMatL=[]; % initialize matrix for groupings by computed modularity values

tic
parpool(cpus) % for parallelizing

parfor ii = 1:N_mod
    % https://sites.google.com/site/bctnet/measures/list#TOC-Clustering-and-Community-Structure
    [tempgnaxL tempmnaxL]=community_louvain(lee_deg2adj);
    GroupsMatL = horzcat(GroupsMatL,tempgnaxL);
    ModMatL = vertcat(ModMatL,tempmnaxL);
    
end

GroupsMatL = GroupsMatL';
toc

delete(gcp) % terminate parallel session

figure;histogram(ModMatL)
ylabel('Counts')
xlabel('Q')

% summary stats for Louvain computed modularity values
ModL_mean = mean(ModMatL);
ModL_median = median(ModMatL);
ModL_mode = mode(ModMatL);
ModL_max = max(ModMatL);
ModL_min = min(ModMatL);
sprintf('mean Louvain Modularity: %.4f', ModL_mean)
sprintf('median Louvain Modularity: %.4f', ModL_median)
sprintf('mode Louvain Modularity: %.4f', ModL_mode)
sprintf('max Louvain Modularity: %.4f', ModL_max)
sprintf('min Louvain Modularity: %.4f', ModL_min)

%% Examine clustering degeneracy and build consensus groups (Empirical)
load('EMidFunctMat.mat')

% https://sites.google.com/site/bctnet/measures/list#TOC-Clustering-and-Community-Structure
dgL = agreement(GroupsMatL');
max_dgL = max(dgL(:));
dg_pL = dgL./max_dgL;

tic
% https://sites.google.com/site/bctnet/measures/list#TOC-Clustering-and-Community-Structure
GroupsL = consensus_und(dg_pL,0.1,1000);
toc

Gl=[lee_deg2ids GroupsL];
% function groups by module, and sorts within group if any function
[ReOrdAMl, ReOrdSKl]=newMod(lee_deg2adj,lee_deg2ids,GroupsL,EMidFunctMat); 

%% Adjacency figure (empirical deg>1)
figure;h = pcolor(ReOrdAMl);
set(h, 'EdgeColor', 'none');

%  manually use colormapeditor and rotate 90 deg
colormap(hot(max(ReOrdAMl(:))+1));
% labels = {'0','1','2','3','4','5','6','7'};
colorbar('location','EastOutside');
title('Consensus Clusters for Louvain')
set(h, 'EdgeColor', 'none');

% outlining clusters
bordBuff = 0.5; 
idClusters=unique(ReOrdSKl(:,2));
nClusters=length(idClusters);

    for i = 1:nClusters
        minBord = find(ReOrdSKl(:,2)==idClusters(i),1,'first') - bordBuff;
        maxBord = find(ReOrdSKl(:,2)==idClusters(i),1,'last') + bordBuff;
        rectangle('Position',[minBord,minBord,maxBord-minBord,maxBord-minBord],...
            'edgecolor','w','LineWidth',1)
    end
    
axis square
set(gcf,'renderer','painters');
colormapeditor

% run fix_pcolor_eps('filename') to turn triangles into squares if necessary


%% Load previously calculated Shuffle variables or see MOD SHUFFLES below
% from degree, weight, and strength preserved shuffles
sW_rand_mod_meanL = load('rand_mod_meanL_n1000_null_dir_sign.mat');
rand_mod_meanL = sW_rand_mod_meanL.rand_mod_meanL;

load('sW_n1000_null_dir_sign.mat')

%% MOD SHUFFLEs (Degree, weight and strength conditioned)
% n_test = 100 and N_Mod = 1000 (Louvain): 11 min on office machine
% n_test = 1000 and N_Mod = 1000 (Louvain): 2 hr on office machine
n_test = 1000 
N_mod = 1000 
binSwaps = 5; % for randmio_*

parpool(cpus) % for parallelizing

rand_mod_minL = zeros(n_test,1);
rand_mod_maxL = zeros(n_test,1);
rand_mod_modeL = zeros(n_test,1);
rand_mod_medianL = zeros(n_test,1);
rand_mod_meanL = zeros(n_test,1);

rand_GroupsMatL = [];
randAdjMat = [];
ccMat = [];
% rand_GroupsNoAxMatL = NaN(N_mod,length(SkelNoInhNoAxNew) * n_test);

tic
parfor i = 1:n_test
    % https://sites.google.com/site/bctnet/null#TOC-Null-network-models
    [randAdj cc] = null_model_dir_sign(lee_deg2adj); % degree, weight, strength preserving shuf

    Shuf_ModMatL = [];
    Shuf_GroupsMatL = [];
    
    randAdjMat=[randAdjMat,randAdj];
    ccMat = [ccMat; cc];

   for ii = 1:N_mod
       % https://sites.google.com/site/bctnet/measures/list#TOC-Clustering-and-Community-Structure
       [tempgnaxL,tempmnaxL]=community_louvain(randAdj);
       Shuf_ModMatL = vertcat(Shuf_ModMatL,tempmnaxL);
       Shuf_GroupsMatL = horzcat(Shuf_GroupsMatL,tempgnaxL);
        
   end
    
   rand_GroupsMatL = [rand_GroupsMatL Shuf_GroupsMatL'];
   
    ModL_median = median(Shuf_ModMatL);
    ModL_mean = mean(Shuf_ModMatL);
    ModL_mode = mode(Shuf_ModMatL);
    ModL_max = max(Shuf_ModMatL);
    ModL_min = min(Shuf_ModMatL);
    
    rand_mod_minL(i,1) = ModL_min;
    rand_mod_maxL(i,1) = ModL_max;
    rand_mod_modeL(i,1) = ModL_mode;
    rand_mod_medianL(i,1) = ModL_median;
    rand_mod_meanL(i,1) = ModL_mean;
    
end
toc

delete(gcp) % terminate parallel session

% for group assignments, making dim1: randomizations and dim3: modularity iterations 
sW = reshape(rand_GroupsMatL,[N_mod,length(lee_deg2ids),n_test]);


%% Histograms of empirical and shuffled Q

figure;histogram(ModMatL,'Normalization','probability','FaceColor','k')
hold on
histogram(rand_mod_meanL,'Normalization','probability','FaceColor',[.5 .5 .5])


ylabel('Probability')
xlabel('Q')
legend('Degree >= 2 Network', 'Deg-, Wgt-, Str- conditioned Shuffle (mean)', 'Location', 'NorthWest');

%% CDFs of empirical and shuffled Q
figure
ecdf(ModMatL)
hold on
ecdf(rand_mod_meanL)


h = get(gca,'children');
set(h(2),'color','k','LineWidth',2);
set(h(1),'color','[.5 .5 .5]','LineWidth',2);


ylabel('Cumulative Probability');
legend('Degree >= 2 Network', 'Degree-conditioned Shuffle (mean)','Location','Northwest');
title('Modularity of Empirical to Degree-conditioned Null');
xlabel('Q');
set(gcf,'renderer','painters')


%% Histograms of empirical vs shuffled Qs
figure
histogram(ModMatL,'Normalization','probability','FaceColor','k')
hold on
histogram(sW_rand_mod_meanL,'Normalization','probability','FaceColor',[.67 .67 .67])
histogram(sD_rand_mod_meanL,'Normalization','probability','FaceColor',[.33 .33 .33])

ylabel('Probability');
legend('Degree >= 2 Network','Degree and Weight-conditioned Shuffle (mean)', 'Degree-conditioned Shuffle (mean)','Location','Northwest');
title('Modularity of Empirical to Conditioned Shuffled Nulls');
xlabel('Q');
set(gcf,'renderer','painters')

% quick stats
[hKS,pKS] = kstest2(ModMatL,sW_rand_mod_meanL)
[pRS,hRS] = ranksum(ModMatL,sW_rand_mod_meanL)

[hKS,pKS] = kstest2(ModMatL,sD_rand_mod_meanL)
[pRS,hRS] = ranksum(ModMatL,sD_rand_mod_meanL)

% %% CDFs of empirical vs shuffled Qs
% figure
% ecdf(ModNoAxMatL)
% hold on
% ecdf(sW_rand_mod_meanL) % degree, weight, strength conditioned
% ecdf(sD_rand_mod_meanL) % degree and out-strength conditioned
% 
% h = get(gca,'children');
% set(h(3),'color','k','LineWidth',2);
% set(h(2),'color','[.67 .67 .67]','LineWidth',2);
% set(h(1),'color','[.33 .33 .33]','LineWidth',2);
% 
% 
% ylabel('Cumulative Probability');
% legend('Degree >= 2 Network','Degree and Weight-conditioned Shuffle (mean)', 'Degree-conditioned Shuffle (mean)','Location','Northwest');
% title('Modularity of Empirical to Conditioned Nulls');
% xlabel('Q');
% set(gcf,'renderer','painters')

% %% for CDFs with shaded Error NOT WORKING
% % TODO: stdev region seems offset
% n_bootstraps = 1000;
% 
% fig = figure;
% hold on
% totbinAllO=[];
% bootbinCO = [];
% bootbinAllO = [];
% bootConvOValues = [];
% bootNonConvOValues = [];
% 
% tic
% 
% for i = 1:n_bootstraps
%     %using randsample to take random sample from ConvD (3rd arg true means
%     %with replacement)
%     %binning x,y pts along the way
%     
%     bootConvO = randsample(ModNoAxMatL,length(ModNoAxMatL),true);
%     bootNonConvO = randsample(rand_mod_meanL,length(rand_mod_meanL),true);
%     
%     %UNCOMMENT THE FOLLOWING IF NOT PLOTTING BAR ALONE
%     kk = cdfplot(bootNonConvO);
%     ll = cdfplot(bootConvO);
% 
%     set(ll,'color',[247,164,164]/255)
%     set(kk,'color',[164,164,247]/255)
% 
%     xCO = get(ll,'XData');
%     xNCO = get(kk,'XData');
% 
%     yCO = get(ll,'YData');
%     yNCO = get(kk,'YData');
% 
%     bootConvOValues = [bootConvOValues;[xCO',yCO']];
%     bootNonConvOValues = [bootNonConvOValues;[xNCO',yNCO']];
% 
%     uistack(ll,'top');
%     uistack(kk,'bottom');
%   
% end
% toc
% 
% %
% bootConvOValues = sortrows(bootConvOValues,1);
% bootNonConvOValues = sortrows(bootNonConvOValues,1);
% 
% yConvOVals = bootConvOValues(:,2);
% yNonConvOVals = bootNonConvOValues(:,2);
% 
% %
% figure;
% 
% cdfplot(ModNoAxMatL)
% hold on
% cdfplot(rand_mod_meanL)
% 
% h = findobj(gca,'type','line');
% set(h(2),'color','k','LineWidth',2);
% set(h(1),'color',[.5 .5 .5],'LineWidth',2);
% 
% ylabel('Cumulative Probability');
% legend('Degree >= 2 Network', 'Degree-conditioned Shuffle (mean)','Location','Northwest');
% title('Modularity of Empirical to Degree-conditioned Null');
% xlabel('Q');
% set(gcf,'renderer','painters')
% 
% Xax = get(gca,'xLim');
% PlotStdevRegionQ(bootConvOValues,bootNonConvOValues,Xax(1),Xax(2));
% % xlim([.46 .58])
% set(gcf,'renderer','painters')
% 

%% p-value of Qmedian and Qmean relative to shuffled modularity values

d_empir_medianL = median(ModMatL);

figure
histogram(rand_mod_medianL,'Normalization','probability')

ylabel('Probability of occurrence')
xlabel('Median Q')
hold on

y = get(gca,'yLim'); % y(2) is the maximum value on the y-axis.
x = get(gca,'xLim'); % x(1) is the minimum value on the x-axis.
plot([d_empir_medianL,d_empir_medianL],y*.99,'r-','lineWidth',2)

% Probability of H0 being true = 
% (# randomly obtained values > observed value)/total number of simulations
p = sum(abs(rand_mod_medianL) > abs(d_empir_medianL))/length(rand_mod_medianL);
text(x(1)+(.01*(abs(x(1))+abs(x(2)))),y(2)*.95,sprintf('H0 is true with %4.4f probability.',p))

% mean
d_empir_meanL = mean(ModMatL);

figure
histogram(rand_mod_meanL,'Normalization','probability')

ylabel('Probability of occurrence')
xlabel('Mean Q')
hold on

y = get(gca,'yLim'); % y(2) is the maximum value on the y-axis.
x = get(gca,'xLim'); % x(1) is the minimum value on the x-axis.
plot([d_empir_meanL,d_empir_meanL],y*.99,'r-','lineWidth',2)

% Probability of H0 being true = 
% (# randomly obtained values > observed value)/total number of simulations
p = sum(abs(rand_mod_meanL) > abs(d_empir_meanL))/length(rand_mod_meanL);
text(x(1)+(.01*(abs(x(1))+abs(x(2)))),y(2)*.95,sprintf('H0 is true with %4.4f probability.',p))


%% Examine clustering degeneracy and build consensus groups (shuffled)
% randID = 3; % select example randomization (1 to n_test)
% randID = randi(n_test) % or randi chosen
[~,randID] = min(abs(rand_mod_meanL-mean(rand_mod_meanL))); % index of closest to mean

% https://sites.google.com/site/bctnet/measures/list#TOC-Clustering-and-Community-Structure
% dgL_shuf = agreement(rand_GroupsNoAxMatL(:,1:201)');
dgL_shuf = agreement(reshape(sW(:,:,randID),[N_mod,length(SkelNoInhNoAxNew)])');

%
max_dgL_shuf = max(dgL_shuf(:));
dg_pL_shuf = dgL_shuf./max_dgL_shuf;

% Consensus clustering (Louvain)
GroupsNoAxL_shuf = consensus_und(dg_pL_shuf,0.1,1000);

GNAXl_shuf=[SkelNoInhNoAxNew GroupsNoAxL_shuf];
[ReOrdNoAxAMl_shuf, ReOrdNoAxSKl_shuf]=newModNoGroupBoundariesOriRGB(AdjNoInhNoAxNew,SkelNoInhNoAxNew,GroupsNoAxL_shuf,EMidOriRGB);


%% Adjacency figure (shuffled deg>1)
figure;h = pcolor(ReOrdNoAxAMl_shuf);
% figure;h = pcolor(rot90(ReOrdNoAxAMl));
set(h, 'EdgeColor', 'none');

% needed to manually use colormapeditor
colormap(hot(max(ReOrdNoAxAMl_shuf(:))+1));
colorbar('location','EastOutside');
title('Consensus Clusters for Louvain (shuffled)')
set(h, 'EdgeColor', 'none');
%view([90 90]); % rotate 90 deg

% outlining clusters
bordBuff = 0.5; 
idClusters=unique(ReOrdNoAxSKl_shuf(:,2));
nClusters=length(idClusters);

    for i = 1:nClusters
        minBord = find(ReOrdNoAxSKl_shuf(:,2)==idClusters(i),1,'first') - bordBuff;
        maxBord = find(ReOrdNoAxSKl_shuf(:,2)==idClusters(i),1,'last') + bordBuff;
        rectangle('Position',[minBord,minBord,maxBord-minBord,maxBord-minBord],...
            'edgecolor','w','LineWidth',1)
    end
    
axis square
set(gcf,'renderer','painters');
colormapeditor

% run fix_pcolor_eps('filename') to turn triangles into squares if
% necessary


%% Empirical to shuffled matrix correlations
% TODO: change corrcoef(ReOrdNoAxAMl(:),ReOrdNoAxAMl_shuf(:)) to
%       corr2(ReOrdNoAxAMl,ReOrdNoAxAMl_shuf)

% in- and out- strength of empirical to randomized matrices
r_in = ccMat(:,1);
r_out = ccMat(:,2);

% empirical to nulls
adjShufMat = [];
r_emp2Shuf = [];

tic
parpool(cpus) % for parallelizing

% parfor ii = 1:100
parfor ii = 1:size(sW,3)

    dgL_shuf = agreement(reshape(sW(:,:,ii),[N_mod,length(SkelNoInhNoAxNew)])');

    max_dgL_shuf = max(dgL_shuf(:));
    dg_pL_shuf = dgL_shuf./max_dgL_shuf;

    % Consensus clustering (Louvain)
    GroupsNoAxL_shuf = consensus_und(dg_pL_shuf,0.1,1000);

    GNAXl_shuf=[SkelNoInhNoAxNew GroupsNoAxL_shuf];
    [ReOrdNoAxAMl_shuf, ReOrdNoAxSKl_shuf]=newModNoGroupBoundariesOriRGB(AdjNoInhNoAxNew,SkelNoInhNoAxNew,GroupsNoAxL_shuf,EMidOriRGB);

    adjShufMat = [adjShufMat, ReOrdNoAxAMl_shuf];
%     r_emp2Shuf = [r_emp2Shuf, corrcoef(ReOrdNoAxAMl(:),ReOrdNoAxAMl_shuf(:))];
    r_emp2Shuf = [r_emp2Shuf, corr2(ReOrdAMl,ReOrdNoAxAMl_shuf)];
    
end

delete(gcp) % terminate parallel session
toc

% r_emp2null = r_emp2Shuf(1,2:2:end)';

adjS = reshape(adjShufMat,[size(adj,1),size(adj,2),n_test]);

% empirical to null matrices
% histograms
figure
hIn = histogram(r_in,'Normalization','probability','Facecolor','blue')
hold on
hOut = histogram(r_out,'Normalization','probability','Facecolor','red')
hNet = histogram(r_emp2null,'Normalization','probability','Facecolor','k')

ylabel('Probability')
xlabel('Correlation Coefficient');
legend('In-strength', 'Out-strength','Network','Location','North');
set(gcf,'renderer','painters')

%% Correlations histograms (separate)
maxY = .25;

figure
hIn = histogram(r_in,'Normalization','probability','Facecolor','blue')
hold on
hOut = histogram(r_out,'Normalization','probability','Facecolor','red')
ylabel('Probability')
xlabel('Correlation Coefficient');
legend('In-strength', 'Out-strength','Location','Northwest');
ylim([0 maxY])
xlim([.8 1])
set(gcf,'renderer','painters')


figure
hNet = histogram(r_emp2null,'Normalization','probability','Facecolor','k')

ylabel('Probability')
xlabel('Correlation Coefficient');
legend('Network','Location','Northeast');
ylim([0 maxY])
xlim([-.01 .19])
set(gcf,'renderer','painters')

%
r_inMean = mean(r_in)
r_outMean = mean(r_out)
r_netMean = mean(r_emp2null)

r_inMedian = median(r_in)
r_outMedian = median(r_out)
r_netMedian = median(r_emp2null)

r_inSter = std(r_in)
r_outSter = std(r_out)
r_netSter = std(r_emp2null)

% %% Correlations line plot
% hIn_binCtrs = [];
% for ii = 1:length(hIn.BinEdges)
%     if ii<length(hIn.BinEdges)
%         hIn_binCtrs = [hIn_binCtrs, (hIn.BinEdges(ii) + hIn.BinEdges(ii+1))/2];
% %         hIn_binCtrs = [hIn_binCtrs, mean(hIn.BinEdges(ii),hIn.BinEdges(ii+1))];
%     end
% end
% 
% hOut_binCtrs = [];
% for ii = 1:length(hOut.BinEdges)
%     if ii<length(hOut.BinEdges)
%         hOut_binCtrs = [hOut_binCtrs, (hOut.BinEdges(ii) + hOut.BinEdges(ii+1))/2];
%     end
% end
% 
% hNet_binCtrs = [];
% for ii = 1:length(hNet.BinEdges)
%     if ii<length(hNet.BinEdges)
%         hNet_binCtrs = [hNet_binCtrs, (hNet.BinEdges(ii) + hNet.BinEdges(ii+1))/2];
%     end
% end
% 
% figure
% plot(hIn_binCtrs,hIn.Values,'b-','lineWidth',2)
% hold on
% plot(hOut_binCtrs,hOut.Values,'r-','lineWidth',2)
% plot(hNet_binCtrs,hNet.Values,'k-','lineWidth',2)
% 
% % Correlations line plot (adjusted to zero)
% figure
% plot([hIn.BinEdges(1),hIn_binCtrs],[0 hIn.Values],'b-','lineWidth',2)
% hold on
% plot([hOut_binCtrs,hOut.BinEdges(end)],[hOut.Values 0],'r-','lineWidth',2)
% plot([hNet.BinEdges(1),hNet_binCtrs],[0 hNet.Values],'k-','lineWidth',2)
% 
% ylabel('Probability')
% xlabel('Correlation Coefficient');
% legend('In-strength', 'Out-strength','Network','Location','North');
% set(gcf,'renderer','painters')
% 
% % Correlations line plot (adjusted to zero, separated)
% figure
% plot([hIn.BinEdges(1),hIn_binCtrs],[0 hIn.Values],'b-','lineWidth',2)
% hold on
% plot([hOut_binCtrs,hOut.BinEdges(end)],[hOut.Values 0],'r-','lineWidth',2)
% ylabel('Probability')
% xlabel('Correlation Coefficient');
% legend('In-strength', 'Out-strength','Location','North');
% set(gcf,'renderer','painters')
% 
% 
% figure
% plot([hNet.BinEdges(1),hNet_binCtrs],[0 hNet.Values],'k-','lineWidth',2)
% 
% ylabel('Probability')
% xlabel('Correlation Coefficient');
% legend('Network','Location','North');
% set(gcf,'renderer','painters')
% 
% %% Correlations CDFs
% figure
% ecdf(r_in)
% hold on
% ecdf(r_out)
% ecdf(r_emp2null)
% 
% 
% h = get(gca,'children');
% set(h(3),'color','blue','LineWidth',2);
% set(h(2),'color','red','LineWidth',2);
% set(h(1),'color','k','LineWidth',2);
% 
% 
% ylabel('Cumulative Probability');
% legend('In-strength', 'Out-strength','Network','Location','North');
% title({'Correlations of Empirical to Null','In-Strengths, Out-Strengths, and Networks'});
% xlabel('Correlation Coefficient');
% set(gcf,'renderer','painters')
% 
% 
% %% Correlations box plots
% figure
% boxplot([r_in(1:100),r_out(1:100),r_emp2null])
% boxplot([r_in,r_out,r_emp2null])


